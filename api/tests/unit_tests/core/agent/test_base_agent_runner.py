import json
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import core.agent.base_agent_runner as module
from core.agent.base_agent_runner import BaseAgentRunner
from graphon.file import FileTransferMethod, FileType
from models.base import TypeBase
from models.enums import ConversationFromSource, CreatorUserRole, MessageFileBelongsTo
from models.model import Message, MessageAgentThought, MessageFile

# ==========================================================
# Fixtures
# ==========================================================


@dataclass(frozen=True)
class _DatabaseBinding:
    session: Session


@pytest.fixture
def agent_session(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    """Bind the runner to a real SQLite session with only its three message tables."""

    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[Message.__table__, MessageAgentThought.__table__, MessageFile.__table__],
    )
    maker = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with maker() as session:
        monkeypatch.setattr(module, "db", _DatabaseBinding(session=session))
        yield session


def _thought(
    *,
    thought_id: str | None = None,
    message_id: str,
    tool: str | None = "tool1;tool2",
    labels: dict[str, object] | None = None,
) -> MessageAgentThought:
    thought = MessageAgentThought(
        message_id=message_id,
        position=1,
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=str(uuid4()),
        thought="",
        tool=tool,
        tool_labels_str=json.dumps(labels or {}),
    )
    if thought_id is not None:
        thought.id = thought_id
    return thought


def _persist_thought(
    session: Session,
    *,
    thought_id: str | None = None,
    message_id: str,
    tool: str | None = "tool1;tool2",
    labels: dict[str, object] | None = None,
) -> MessageAgentThought:
    thought = _thought(
        thought_id=thought_id,
        message_id=message_id,
        tool=tool,
        labels=labels,
    )
    session.add(thought)
    session.commit()
    return thought


def _persist_message(session: Session, *, message_id: str, conversation_id: str, answer: str = "") -> Message:
    message = Message(
        id=message_id,
        app_id=str(uuid4()),
        conversation_id=conversation_id,
        query="hello",
        message={},
        message_unit_price=Decimal(0),
        answer=answer,
        answer_unit_price=Decimal(0),
        currency="USD",
        from_source=ConversationFromSource.CONSOLE,
    )
    message._inputs = {}
    session.add(message)
    session.commit()
    return message


def _persist_message_file(session: Session, *, message_id: str) -> MessageFile:
    message_file = MessageFile(
        message_id=message_id,
        type=FileType.IMAGE,
        transfer_method=FileTransferMethod.REMOTE_URL,
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=str(uuid4()),
        belongs_to=MessageFileBelongsTo.USER,
        url="https://example.test/image.png",
    )
    session.add(message_file)
    session.commit()
    return message_file


def _prepare_history(
    session: Session,
    runner: BaseAgentRunner,
    mocker: MockerFixture,
    returned_messages: list[object],
) -> None:
    _persist_message(
        session,
        message_id=str(uuid4()),
        conversation_id=runner.message.conversation_id,
    )
    mocker.patch.object(module, "extract_thread_messages", return_value=returned_messages)


@pytest.fixture
def runner(mocker: MockerFixture, agent_session: Session):
    r = BaseAgentRunner.__new__(BaseAgentRunner)
    r.tenant_id = str(uuid4())
    r.user_id = str(uuid4())
    r.agent_thought_count = 0
    r.message = mocker.MagicMock(id=str(uuid4()), conversation_id=str(uuid4()))
    r.app_config = mocker.MagicMock()
    r.app_config.app_id = "app1"
    r.app_config.agent = None
    r.dataset_tools = []
    r.application_generate_entity = mocker.MagicMock(invoke_from="test")
    r._current_thoughts = []
    return r


# ==========================================================
# _repack_app_generate_entity
# ==========================================================


class TestRepack:
    def test_sets_empty_if_none(self, runner: BaseAgentRunner, mocker: MockerFixture):
        entity = mocker.MagicMock()
        entity.app_config.prompt_template.simple_prompt_template = None
        result = runner._repack_app_generate_entity(entity)
        assert result.app_config.prompt_template.simple_prompt_template == ""

    def test_keeps_existing(self, runner: BaseAgentRunner, mocker: MockerFixture):
        entity = mocker.MagicMock()
        entity.app_config.prompt_template.simple_prompt_template = "abc"
        result = runner._repack_app_generate_entity(entity)
        assert result.app_config.prompt_template.simple_prompt_template == "abc"


# ==========================================================
# update_prompt_message_tool
# ==========================================================


class TestUpdatePromptTool:
    def test_replaces_prompt_tool_parameters_with_tool_schema(self, runner: BaseAgentRunner, mocker: MockerFixture):
        tool = mocker.MagicMock()
        schema = {
            "type": "object",
            "properties": {"p1": {"type": "string", "description": "desc"}},
            "required": ["p1"],
        }
        tool.get_llm_parameters_json_schema.return_value = schema

        prompt_tool = mocker.MagicMock()
        prompt_tool.parameters = {"properties": {}, "required": []}

        result = runner.update_prompt_message_tool(tool, prompt_tool)
        assert result.parameters == schema


# ==========================================================
# create_agent_thought
# ==========================================================


class TestCreateAgentThought:
    def test_with_files(self, runner: BaseAgentRunner, agent_session: Session):
        message_id = str(uuid4())

        result = runner.create_agent_thought(message_id, "msg", "tool", "input", ["f1"])

        thought = agent_session.get_one(MessageAgentThought, result)
        assert thought.message_id == message_id
        assert thought.message_files == json.dumps(["f1"])
        assert runner.agent_thought_count == 1

    def test_without_files(self, runner: BaseAgentRunner, agent_session: Session):
        message_id = str(uuid4())

        result = runner.create_agent_thought(message_id, "msg", "tool", "input", [])

        assert agent_session.get_one(MessageAgentThought, result).message_files == ""


# ==========================================================
# save_agent_thought
# ==========================================================


class TestSaveAgentThought:
    def test_not_found(self, runner: BaseAgentRunner, agent_session: Session):
        with pytest.raises(ValueError):
            runner.save_agent_thought(str(uuid4()), None, None, None, None, None, None, [], None)

    def test_full_update(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        agent = _persist_thought(agent_session, message_id=str(uuid4()))

        mock_label = mocker.MagicMock()
        mock_label.to_dict.return_value = {"en_US": "label"}
        mocker.patch.object(module.ToolManager, "get_tool_label", return_value=mock_label)

        usage = mocker.MagicMock(
            prompt_tokens=1,
            prompt_price_unit=Decimal("0.1"),
            prompt_unit_price=Decimal("0.1"),
            completion_tokens=2,
            completion_price_unit=Decimal("0.2"),
            completion_unit_price=Decimal("0.2"),
            total_tokens=3,
            total_price=Decimal("0.3"),
        )

        runner.save_agent_thought(
            agent.id,
            "tool1;tool2",
            {"a": 1},
            "thought",
            {"b": 2},
            {"meta": 1},
            "answer",
            ["f1"],
            usage,
        )

        persisted = agent_session.get_one(MessageAgentThought, agent.id)
        assert persisted.answer == "answer"
        assert persisted.tokens == 3
        assert "tool1" in json.loads(persisted.tool_labels_str)

    def test_label_fallback_when_none(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        agent = _persist_thought(agent_session, message_id=str(uuid4()), tool="unknown_tool")
        mocker.patch.object(module.ToolManager, "get_tool_label", return_value=None)

        runner.save_agent_thought(agent.id, None, None, None, None, None, None, [], None)
        labels = json.loads(agent_session.get_one(MessageAgentThought, agent.id).tool_labels_str)
        assert "unknown_tool" in labels

    def test_messages_ids_none(self, runner: BaseAgentRunner, agent_session: Session):
        agent = _persist_thought(agent_session, message_id=str(uuid4()))

        runner.save_agent_thought(agent.id, None, None, None, None, None, None, None, None)

        assert agent_session.get_one(MessageAgentThought, agent.id).message_files is None

    def test_success_dict_serialization(self, runner: BaseAgentRunner, agent_session: Session):
        agent = _persist_thought(agent_session, message_id=str(uuid4()))

        runner.save_agent_thought(
            agent.id,
            None,
            {"a": 1},
            None,
            {"b": 2},
            None,
            None,
            [],
            None,
        )

        persisted = agent_session.get_one(MessageAgentThought, agent.id)
        assert isinstance(persisted.tool_input, str)
        assert isinstance(persisted.observation, str)


# ==========================================================
# organize_agent_user_prompt
# ==========================================================


class TestOrganizeUserPrompt:
    def test_no_files(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        msg = mocker.MagicMock(id="1", query="hello", app_model_config=None)
        result = runner.organize_agent_user_prompt(msg)
        assert result.content == "hello"

    def test_with_files_no_config(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        message_id = str(uuid4())
        _persist_message_file(agent_session, message_id=message_id)
        msg = mocker.MagicMock(id=message_id, query="hello", app_model_config=None)
        result = runner.organize_agent_user_prompt(msg)
        assert result.content == "hello"

    def test_image_detail_low_fallback(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        message_id = str(uuid4())
        _persist_message_file(agent_session, message_id=message_id)
        file_config = mocker.MagicMock()
        file_config.image_config = mocker.MagicMock(detail=None)
        mocker.patch.object(module.FileUploadConfigManager, "convert", return_value=file_config)
        mocker.patch.object(module.file_factory, "build_from_message_files", return_value=[])

        msg = mocker.MagicMock(id=message_id, query="hello")
        msg.app_model_config.to_dict.return_value = {}

        result = runner.organize_agent_user_prompt(msg)
        assert result.content == "hello"


# ==========================================================
# organize_agent_history
# ==========================================================


class TestOrganizeHistory:
    def test_empty(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        mocker.patch.object(module, "extract_thread_messages", return_value=[])
        result = runner.organize_agent_history([])
        assert result == []

    def test_with_answer_only(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        msg = mocker.MagicMock(id="m1", answer="ans", agent_thoughts=[], app_model_config=None)
        _prepare_history(agent_session, runner, mocker, [msg])
        result = runner.organize_agent_history([])
        assert any(isinstance(x, module.AssistantPromptMessage) for x in result)

    def test_skip_current_message(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        msg = mocker.MagicMock(id=runner.message.id, agent_thoughts=[], answer="ans", app_model_config=None)
        _prepare_history(agent_session, runner, mocker, [msg])
        result = runner.organize_agent_history([])
        assert result == []

    def test_with_tool_calls_invalid_json(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        thought = mocker.MagicMock(
            tool="tool1",
            tool_input="invalid",
            observation="invalid",
            thought="thinking",
        )
        msg = mocker.MagicMock(id="m2", agent_thoughts=[thought], answer=None, app_model_config=None)

        _prepare_history(agent_session, runner, mocker, [msg])
        mocker.patch("uuid.uuid4", return_value="uuid")

        result = runner.organize_agent_history([])
        assert isinstance(result, list)

    def test_empty_tool_name_split(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        thought = mocker.MagicMock(tool=";", thought="thinking")
        msg = mocker.MagicMock(id="m5", agent_thoughts=[thought], answer=None, app_model_config=None)

        _prepare_history(agent_session, runner, mocker, [msg])
        result = runner.organize_agent_history([])
        assert isinstance(result, list)

    def test_valid_json_tool_flow(self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture):
        thought = mocker.MagicMock(
            tool="tool1",
            tool_input=json.dumps({"tool1": {"x": 1}}),
            observation=json.dumps({"tool1": "obs"}),
            thought="thinking",
        )

        msg = mocker.MagicMock(
            id="m100",
            agent_thoughts=[thought],
            answer=None,
            app_model_config=None,
        )

        _prepare_history(agent_session, runner, mocker, [msg])
        mocker.patch("uuid.uuid4", return_value="uuid")

        result = runner.organize_agent_history([])
        assert isinstance(result, list)


# ==========================================================
# _convert_tool_to_prompt_message_tool (new coverage)
# ==========================================================


class TestConvertToolToPromptMessageTool:
    def test_basic_conversion(self, runner: BaseAgentRunner, mocker: MockerFixture):
        tool = mocker.MagicMock(tool_name="tool1")

        tool_entity = mocker.MagicMock()
        tool_entity.entity.description.llm = "desc"
        schema = {
            "type": "object",
            "properties": {"param1": {"type": "string", "description": "desc"}},
            "required": ["param1"],
        }
        tool_entity.get_llm_parameters_json_schema.return_value = schema

        mocker.patch.object(module.ToolManager, "get_agent_tool_runtime", return_value=tool_entity)
        mocker.patch.object(module, "PromptMessageTool", side_effect=lambda **kw: MagicMock(**kw))

        prompt_tool, entity = runner._convert_tool_to_prompt_message_tool(tool)
        assert entity == tool_entity
        assert prompt_tool.parameters == schema


# ==========================================================
# _init_prompt_tools additional branches
# ==========================================================


class TestInitPromptToolsExtended:
    def test_agent_tool_branch(self, runner: BaseAgentRunner, mocker: MockerFixture):
        agent_tool = mocker.MagicMock(tool_name="agent_tool")
        runner.app_config.agent = mocker.MagicMock(tools=[agent_tool])
        mocker.patch.object(runner, "_convert_tool_to_prompt_message_tool", return_value=(MagicMock(), "entity"))

        tools, prompts = runner._init_prompt_tools()
        assert "agent_tool" in tools

    def test_exception_in_conversion(self, runner: BaseAgentRunner, mocker: MockerFixture):
        agent_tool = mocker.MagicMock(tool_name="bad_tool")
        runner.app_config.agent = mocker.MagicMock(tools=[agent_tool])
        mocker.patch.object(runner, "_convert_tool_to_prompt_message_tool", side_effect=Exception)

        tools, prompts = runner._init_prompt_tools()
        assert tools == {}


# ==========================================================
# Additional Coverage Tests (DO NOT MODIFY EXISTING TESTS)
# ==========================================================


class TestAdditionalCoverage:
    def test_save_agent_thought_existing_labels(self, runner: BaseAgentRunner, agent_session: Session):
        agent = _persist_thought(
            agent_session,
            message_id=str(uuid4()),
            tool="tool1",
            labels={"tool1": {"en_US": "existing"}},
        )

        runner.save_agent_thought(agent.id, None, None, None, None, None, None, [], None)
        labels = json.loads(agent_session.get_one(MessageAgentThought, agent.id).tool_labels_str)
        assert labels["tool1"]["en_US"] == "existing"

    def test_save_agent_thought_tool_meta_string(self, runner: BaseAgentRunner, agent_session: Session):
        agent = _persist_thought(agent_session, message_id=str(uuid4()), tool="tool1")

        runner.save_agent_thought(agent.id, None, None, None, None, "meta_string", None, [], None)
        assert agent_session.get_one(MessageAgentThought, agent.id).tool_meta_str == "meta_string"

    def test_convert_dataset_retriever_tool(self, runner: BaseAgentRunner, mocker: MockerFixture):
        ds_tool = mocker.MagicMock()
        ds_tool.entity.identity.name = "ds"
        ds_tool.entity.description.llm = "desc"

        param = mocker.MagicMock()
        param.name = "query"
        param.llm_description = "desc"
        param.required = True

        ds_tool.get_runtime_parameters.return_value = [param]

        mocker.patch.object(module, "PromptMessageTool", side_effect=lambda **kw: MagicMock(**kw))

        prompt = runner._convert_dataset_retriever_tool_to_prompt_message_tool(ds_tool)
        assert prompt is not None

    def test_organize_user_prompt_with_file_objects(
        self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture
    ):
        message_id = str(uuid4())
        _persist_message_file(agent_session, message_id=message_id)

        file_config = mocker.MagicMock()
        file_config.image_config = mocker.MagicMock(detail=None)

        mocker.patch.object(module.FileUploadConfigManager, "convert", return_value=file_config)
        mocker.patch.object(module.file_factory, "build_from_message_files", return_value=["file1"])
        mocker.patch.object(module.file_manager, "to_prompt_message_content", return_value=mocker.MagicMock())

        mocker.patch.object(module, "UserPromptMessage", side_effect=lambda **kw: MagicMock(**kw))
        mocker.patch.object(module, "TextPromptMessageContent", side_effect=lambda **kw: MagicMock(**kw))

        msg = mocker.MagicMock(id=message_id, query="hello")
        msg.app_model_config.to_dict.return_value = {}

        result = runner.organize_agent_user_prompt(msg)
        assert result is not None

    def test_organize_history_without_tool_names(
        self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture
    ):
        thought = mocker.MagicMock(tool=None, thought="thinking")
        msg = mocker.MagicMock(id="m3", agent_thoughts=[thought], answer=None, app_model_config=None)

        _prepare_history(agent_session, runner, mocker, [msg])

        result = runner.organize_agent_history([])
        assert isinstance(result, list)

    def test_organize_history_multiple_tools_split(
        self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture
    ):
        thought = mocker.MagicMock(
            tool="tool1;tool2",
            tool_input=json.dumps({"tool1": {}, "tool2": {}}),
            observation=json.dumps({"tool1": "o1", "tool2": "o2"}),
            thought="thinking",
        )
        msg = mocker.MagicMock(id="m4", agent_thoughts=[thought], answer=None, app_model_config=None)

        _prepare_history(agent_session, runner, mocker, [msg])
        mocker.patch("uuid.uuid4", return_value="uuid")

        result = runner.organize_agent_history([])
        assert isinstance(result, list)


class TestConvertDatasetRetrieverTool:
    def test_required_param_added(self, runner: BaseAgentRunner, mocker: MockerFixture):
        ds_tool = mocker.MagicMock()
        ds_tool.entity.identity.name = "ds"
        ds_tool.entity.description.llm = "desc"

        param = mocker.MagicMock()
        param.name = "query"
        param.llm_description = "desc"
        param.required = True

        ds_tool.get_runtime_parameters.return_value = [param]

        mocker.patch.object(module, "PromptMessageTool", side_effect=lambda **kw: MagicMock(**kw))

        prompt = runner._convert_dataset_retriever_tool_to_prompt_message_tool(ds_tool)

        assert prompt is not None


class TestBaseAgentRunnerInit:
    def test_init_sets_stream_tool_call_and_files(self, agent_session: Session, mocker: MockerFixture):
        message_id = str(uuid4())
        _persist_thought(agent_session, message_id=message_id)
        second = _thought(message_id=message_id)
        second.position = 2
        agent_session.add(second)
        agent_session.commit()

        mocker.patch.object(BaseAgentRunner, "organize_agent_history", return_value=[])
        mocker.patch.object(module.DatasetRetrieverTool, "get_dataset_tools", return_value=["ds_tool"])

        llm = mocker.MagicMock()
        llm.get_model_schema.return_value = mocker.MagicMock(
            features=[module.ModelFeature.STREAM_TOOL_CALL, module.ModelFeature.VISION]
        )
        model_instance = mocker.MagicMock(model_type_instance=llm, model="m", credentials="c")

        app_config = mocker.MagicMock()
        app_config.app_id = "app1"
        app_config.agent = None
        app_config.dataset = mocker.MagicMock(dataset_ids=["d1"], retrieve_config={"k": "v"})
        app_config.additional_features = mocker.MagicMock(show_retrieve_source=True)

        app_generate = mocker.MagicMock(invoke_from="test", inputs={}, files=["file1"])
        message = mocker.MagicMock(id=message_id, conversation_id=str(uuid4()))

        runner = BaseAgentRunner(
            session=agent_session,
            tenant_id="tenant",
            application_generate_entity=app_generate,
            conversation=mocker.MagicMock(),
            app_config=app_config,
            model_config=mocker.MagicMock(),
            config=mocker.MagicMock(),
            queue_manager=mocker.MagicMock(),
            message=message,
            user_id="user",
            model_instance=model_instance,
        )

        assert runner.stream_tool_call is True
        assert runner.files == ["file1"]
        assert runner.dataset_tools == ["ds_tool"]
        assert runner.agent_thought_count == 2


class TestBaseAgentRunnerCoverage:
    def test_init_prompt_tools_adds_dataset_tools(self, runner: BaseAgentRunner, mocker: MockerFixture):
        dataset_tool = mocker.MagicMock()
        dataset_tool.entity.identity.name = "ds"
        runner.dataset_tools = [dataset_tool]

        mocker.patch.object(runner, "_convert_dataset_retriever_tool_to_prompt_message_tool", return_value=MagicMock())

        tools, prompt_tools = runner._init_prompt_tools()

        assert tools["ds"] == dataset_tool
        assert len(prompt_tools) == 1

    def test_save_agent_thought_json_dumps_fallbacks(
        self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture
    ):
        agent = _persist_thought(agent_session, message_id=str(uuid4()), tool="tool1")

        mocker.patch.object(module.ToolManager, "get_tool_label", return_value=None)

        tool_input = {"a": 1}
        observation = {"b": 2}
        tool_meta = {"c": 3}

        real_dumps = json.dumps

        def dumps_side_effect(value, *args, **kwargs):
            if value in (tool_input, observation, tool_meta) and kwargs.get("ensure_ascii") is False:
                raise TypeError("fail")
            return real_dumps(value, *args, **kwargs)

        mocker.patch.object(module.json, "dumps", side_effect=dumps_side_effect)

        runner.save_agent_thought(
            agent.id,
            "tool1",
            tool_input,
            None,
            observation,
            tool_meta,
            None,
            [],
            None,
        )

        persisted = agent_session.get_one(MessageAgentThought, agent.id)
        assert isinstance(persisted.tool_input, str)
        assert isinstance(persisted.observation, str)
        assert isinstance(persisted.tool_meta_str, str)

    def test_save_agent_thought_skips_empty_tool_name(
        self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture
    ):
        agent = _persist_thought(agent_session, message_id=str(uuid4()), tool="tool1;;")

        mocker.patch.object(module.ToolManager, "get_tool_label", return_value=None)

        runner.save_agent_thought(agent.id, None, None, None, None, None, None, [], None)

        labels = json.loads(agent_session.get_one(MessageAgentThought, agent.id).tool_labels_str)
        assert "" not in labels

    def test_organize_history_includes_system_prompt(
        self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture
    ):
        mocker.patch.object(module, "extract_thread_messages", return_value=[])

        system_message = module.SystemPromptMessage(content="sys")

        result = runner.organize_agent_history([system_message])

        assert system_message in result

    def test_organize_history_tool_inputs_and_observation_none(
        self, runner: BaseAgentRunner, agent_session: Session, mocker: MockerFixture
    ):
        thought = mocker.MagicMock(
            tool="tool1",
            tool_input=None,
            observation=None,
            thought="thinking",
        )
        msg = mocker.MagicMock(id="m6", agent_thoughts=[thought], answer=None, app_model_config=None)

        _prepare_history(agent_session, runner, mocker, [msg])
        mocker.patch("uuid.uuid4", return_value="uuid")

        mocker.patch.object(
            runner,
            "organize_agent_user_prompt",
            return_value=module.UserPromptMessage(content="user"),
        )

        result = runner.organize_agent_history([])

        assert any(isinstance(item, module.ToolPromptMessage) for item in result)

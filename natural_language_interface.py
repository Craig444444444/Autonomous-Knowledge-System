import logging
import re
import json
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from enum import Enum, auto

LOGGER = logging.getLogger("aks")

class CommandType(Enum):
    """Enumeration of supported command types."""
    GENERATE_CODE = auto()
    GENERATE_DATA = auto()
    CREATE_BRANCH = auto()
    CHECKOUT_BRANCH = auto()
    MERGE_BRANCH = auto()
    DELETE_BRANCH = auto()
    CREATE_SNAPSHOT = auto()
    RESTORE_SNAPSHOT = auto()
    RESEARCH_TOPIC = auto()
    ENHANCE_CODEBASE = auto()
    RUN_SHELL_COMMAND = auto()
    SHUTDOWN = auto()
    PROCESS_FEEDBACK = auto()
    AI_INTERPRETED = auto()
    UNKNOWN = auto()

@dataclass
class CommandResult:
    """Structured result of command parsing."""
    type: CommandType
    raw_text: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    source: str = "pattern"
    validation_errors: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.name.lower(),
            "raw": self.raw_text,
            "params": self.parameters,
            "confidence": self.confidence,
            "source": self.source,
            "valid": not self.validation_errors,
            "errors": self.validation_errors or []
        }

class NaturalLanguageInterface:
    """
    Enhanced natural language command processor with improved parsing,
    validation, and security checks.
    """
    def __init__(self, ai_manager: Any):
        """
        Initialize the Natural Language Interface with enhanced capabilities.

        Args:
            ai_manager: Reference to the AI provider manager.
        """
        self.ai_manager = ai_manager
        self._pattern_cache = {}
        self._last_ai_call = 0
        self._rate_limit = 2.0  # seconds between AI calls
        self._init_patterns()
        LOGGER.info("Natural Language Interface initialized with enhanced parser")

    def _init_patterns(self):
        """Initialize and compile all regex patterns for efficiency."""
        self._patterns = [
            # (pattern, command_type, parameter_groups, validation_func)
            (
                r"generate\s+(?:python\s+)?code\s+(?:for|to|that)\s+(.+)",
                CommandType.GENERATE_CODE,
                {"prompt": 1},
                self._validate_generation_params
            ),
            (
                r"generate\s+(?:test\s+)?data\s+(?:for|about|on)\s+(.+)",
                CommandType.GENERATE_DATA,
                {"prompt": 1},
                self._validate_generation_params
            ),
            (
                r"(?:create|make)\s+(?:new\s+)?branch\s+(?:named\s*)?['\"]?([a-zA-Z0-9/_-]+)['\"]?",
                CommandType.CREATE_BRANCH,
                {"name": 1},
                self._validate_branch_name
            ),
            (
                r"(?:checkout|switch\s+to)\s+branch\s+(?:['\"]?)([a-zA-Z0-9/_-]+)(?:['\"]?)",
                CommandType.CHECKOUT_BRANCH,
                {"name": 1},
                self._validate_branch_name
            ),
            (
                r"merge\s+branch\s+(?:['\"]?)([a-zA-Z0-9/-]+)(?:['\"]?)\s+into\s+(?:['\"]?)([a-zA-Z0-9/-]+)(?:['\"]?)",
                CommandType.MERGE_BRANCH,
                {"source": 1, "target": 2},
                self._validate_merge_params
            ),
            (
                r"(?:delete|remove)\s+branch\s+(?:['\"]?)([a-zA-Z0-9/_-]+)(?:['\"]?)",
                CommandType.DELETE_BRANCH,
                {"name": 1},
                self._validate_branch_name
            ),
            (
                r"create\s+snapshot\s+(?:with\s+tag\s*)?(?:['\"]?)([a-zA-Z0-9_-]+)(?:['\"]?)",
                CommandType.CREATE_SNAPSHOT,
                {"tag": 1},
                self._validate_snapshot_tag
            ),
            (
                r"restore\s+snapshot\s+(?:from\s*)?(?:['\"]?)([a-zA-Z0-9_.-]+\.zip|latest)(?:['\"]?)",
                CommandType.RESTORE_SNAPSHOT,
                {"snapshot": 1},
                self._validate_snapshot_name
            ),
            (
                r"(?:research|find|look\s+up)\s+(?:info\s+on\s)?(.+)",
                CommandType.RESEARCH_TOPIC,
                {"query": 1},
                self._validate_research_query
            ),
            (
                r"(?:enhance|improve|refactor)\s+(?:the\s+)?codebase\s+(?:for|to)\s+(.+)",
                CommandType.ENHANCE_CODEBASE,
                {"goal": 1},
                self._validate_enhancement_goal
            ),
            (
                r"(?:run|execute)\s+(?:shell\s+)?command:\s+(.+)",
                CommandType.RUN_SHELL_COMMAND,
                {"command": 1},
                self._validate_shell_command
            ),
            (
                r"(?:shutdown|exit|terminate|stop)",
                CommandType.SHUTDOWN,
                {},
                None
            ),
            (
                r"(?:process|handle)\s+(?:user\s+)?feedback(?: zip)?",
                CommandType.PROCESS_FEEDBACK,
                {},
                None
            )
        ]

        # Compile regex patterns for better performance
        self._compiled_patterns = []
        for pattern, cmd_type, params, validator in self._patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._compiled_patterns.append(
                    (compiled, cmd_type, params, validator)
                )
            except re.error as e:
                LOGGER.error(f"Failed to compile regex pattern '{pattern}': {e}")

    def parse_command(self, command_text: str) -> Dict[str, Any]:
        """
        Parse and validate a natural language command with enhanced capabilities.

        Args:
            command_text: Raw input command string

        Returns:
            Dictionary containing structured command information
        """
        command_text = command_text.strip()
        if not command_text:
            return self._unknown_command("Empty command")

        # First try fast pattern matching
        start_time = time.time()
        parsed = self._parse_with_patterns(command_text)

        # Fall back to AI if needed (with rate limiting)
        if parsed.type == CommandType.UNKNOWN and self._can_use_ai():
            LOGGER.debug(f"Falling back to AI for command: {command_text[:50]}...")
            ai_parsed = self._parse_with_ai(command_text)
            if ai_parsed and ai_parsed.type != CommandType.UNKNOWN:
                parsed = ai_parsed
                self._last_ai_call = time.time()

        # Validate the parsed command
        if parsed.validation_errors is None:
            parsed.validation_errors = self._validate_command(parsed)

        LOGGER.debug(
            f"Parsed command in {(time.time() - start_time)*1000:.2f}ms: "
            f"{parsed.to_dict()}"
        )
        return parsed.to_dict()

    def _parse_with_patterns(self, command_text: str) -> CommandResult:
        """Parse command using optimized pattern matching."""
        cmd_lower = command_text.lower()
        best_match = None
        best_score = 0

        for pattern, cmd_type, params, _ in self._compiled_patterns:
            match = pattern.search(cmd_lower)
            if match:
                # Score matches by length of matched text
                score = match.end() - match.start()
                if score > best_score:
                    best_score = score
                    parameters = {
                        key: match.group(idx).strip()
                        for key, idx in params.items()
                        if match.group(idx)
                    }
                    best_match = CommandResult(
                        type=cmd_type,
                        raw_text=command_text,
                        parameters=parameters,
                        confidence=min(1.0, score / len(command_text)),
                        source="pattern"
                    )

        return best_match or self._unknown_command(command_text)

    def _parse_with_ai(self, command_text: str) -> Optional[CommandResult]:
        """Parse command using AI with enhanced prompt engineering."""
        system_prompt = """You are a command parsing assistant. Analyze the user's command and
return a JSON object with these fields:
"type": The command type (from the allowed list)
"parameters": Key-value pairs of extracted parameters
"confidence": Your confidence score (0.0-1.0)

Allowed command types:
["generate_code", "generate_data", "create_branch", "checkout_branch",
"merge_branch", "delete_branch", "create_snapshot", "restore_snapshot",
"research_topic", "enhance_codebase", "run_shell_command", "shutdown",
"process_feedback", "ai_interpreted"]

Return ONLY the JSON object with no additional text or explanation."""

        user_prompt = f"""User command to parse: "{command_text}"

Extract the command type and parameters. For example:
"write a function to sort a list" =>
  {{"type": "generate_code", "parameters": {{"prompt": "function to sort a list"}}, "confidence": 0.95}}
"make a branch called feature-123" =>
  {{"type": "create_branch", "parameters": {{"name": "feature-123"}}, "confidence": 0.99}}"""

        try:
            response = self.ai_manager.generate_text(
                user_prompt,
                system_prompt,
                max_tokens=500
            )
            if not response:
                return None

            # Extract JSON from response
            json_str = self._extract_json(response)
            if not json_str:
                LOGGER.warning("AI response contained no valid JSON")
                return None

            data = json.loads(json_str)
            if not isinstance(data, dict):
                raise ValueError("AI response was not a JSON object")

            # Convert to CommandResult
            try:
                cmd_type = CommandType[data["type"].upper()]
            except KeyError:
                cmd_type = CommandType.UNKNOWN

            return CommandResult(
                type=cmd_type,
                raw_text=command_text,
                parameters=data.get("parameters", {}),
                confidence=float(data.get("confidence", 0.0)),
                source="ai"
            )

        except json.JSONDecodeError as e:
            LOGGER.error(f"Failed to decode AI response: {e}\nResponse: {response}")
        except Exception as e:
            LOGGER.error(f"AI parsing failed: {e}", exc_info=True)
        return None

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract the first valid JSON object from text."""
        # First try to parse the entire response
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # If that fails, look for a JSON object within the text
        matches = re.findall(r'\{.*\}', text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        return None

    def _validate_command(self, command: CommandResult) -> List[str]:
        """Validate a parsed command against its requirements."""
        if command.type == CommandType.UNKNOWN:
            return ["Unknown command type"]

        # Find the validator for this command type
        validator = None
        for _, cmd_type, _, v in self._patterns:
            if cmd_type == command.type:
                validator = v
                break

        if not validator:
            return []

        try:
            return validator(command.parameters) or []
        except Exception as e:
            LOGGER.error(f"Validation failed for {command.type}: {e}")
            return [f"Validation error: {str(e)}"]

    def _validate_generation_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for generation commands."""
        errors = []
        if "prompt" not in params or len(params["prompt"]) < 5:
            errors.append("Prompt must be at least 5 characters")
        return errors

    def _validate_branch_name(self, params: Dict[str, Any]) -> List[str]:
        """Validate branch name parameters."""
        errors = []
        name = params.get("name", "")
        if not name:
            errors.append("Branch name cannot be empty")
        elif not re.match(r"^[a-zA-Z0-9_\-/]+$", name):
            errors.append("Branch name contains invalid characters")
        elif len(name) > 100:
            errors.append("Branch name too long (max 100 chars)")
        return errors

    def _validate_merge_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate branch merge parameters."""
        errors = []
        if params.get("source") == params.get("target"):
            errors.append("Cannot merge branch into itself")
        return errors + self._validate_branch_name({"name": params.get("source")}) + \
               self._validate_branch_name({"name": params.get("target")})

    def _validate_snapshot_tag(self, params: Dict[str, Any]) -> List[str]:
        """Validate snapshot tag parameters."""
        tag = params.get("tag", "")
        if not tag:
            return ["Snapshot tag cannot be empty"]
        if not re.match(r"^[a-zA-Z0-9_\-]+$", tag):
            return ["Snapshot tag contains invalid characters"]
        return []

    def _validate_snapshot_name(self, params: Dict[str, Any]) -> List[str]:
        """Validate snapshot name parameters."""
        name = params.get("snapshot", "")
        if not name:
            return ["Snapshot name cannot be empty"]
        if name != "latest" and not name.endswith(".zip"):
            return ["Snapshot must be 'latest' or a .zip filename"]
        return []

    def _validate_research_query(self, params: Dict[str, Any]) -> List[str]:
        """Validate research query parameters."""
        query = params.get("query", "")
        if not query or len(query) < 3:
            return ["Research query must be at least 3 characters"]
        return []

    def _validate_enhancement_goal(self, params: Dict[str, Any]) -> List[str]:
        """Validate codebase enhancement parameters."""
        goal = params.get("goal", "")
        if not goal or len(goal) < 5:
            return ["Enhancement goal must be at least 5 characters"]
        return []

    def _validate_shell_command(self, params: Dict[str, Any]) -> List[str]:
        """Validate shell command parameters with security checks."""
        cmd = params.get("command", "")
        if not cmd:
            return ["Command cannot be empty"]

        # Security checks
        errors = []
        dangerous_patterns = [
            (r"[;&|`]", "Multiple command execution"),
            (r"\$\(", "Command substitution"),
            (r">\s*", "File redirection"),
            (r"rm\s+-[rf]", "Forced recursive deletion"),
            (r"/dev/null", "Output suppression"),
            (r"wget\s+http", "Remote code download"),
            (r"curl\s+http", "Remote code download"),
            (r"chmod\s+[0-7]{3,4}", "Permission modification"),
            (r"^sudo", "Root privilege escalation")
        ]

        for pattern, msg in dangerous_patterns:
            if re.search(pattern, cmd):
                errors.append(f"Potentially dangerous command: {msg}")

        return errors

    def _unknown_command(self, text: str) -> CommandResult:
        """Create result for unknown commands."""
        return CommandResult(
            type=CommandType.UNKNOWN,
            raw_text=text,
            parameters={},
            confidence=0.0,
            source="none"
        )

    def _can_use_ai(self) -> bool:
        """Check if AI parsing is available and not rate-limited."""
        if not self.ai_manager.has_available_providers():
            return False
        return (time.time() - self._last_ai_call) >= self._rate_limit

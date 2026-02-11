from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Any
from langchain_openai import ChatOpenAI
from src.config.settings import get_settings
from src.utils.logger import get_logger

StateT = TypeVar('StateT')

class BaseAgent(ABC, Generic[StateT]):
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self.llm = llm or ChatOpenAI(
            model_name=self.settings.OPENAI_MODEL,
            temperature=self.settings.OPENAI_TEMPERATURE,
            api_key=self.settings.OPENAI_API_KEY
        )
    
    @abstractmethod
    def execute(self, state: StateT) -> StateT:
        """Execute agent logic"""
        pass
    
    def _log_execution(self, state: Any) -> None:
        """Log agent execution for debugging"""
        self.logger.info(f"Executing {self.__class__.__name__}")

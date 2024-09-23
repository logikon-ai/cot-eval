"""Config Class for COT evaluations"""

from typing import Optional

from pydantic import BaseModel
import yaml

class COTEvalConfig(BaseModel):
    """Config Class for COT evaluations"""

    name: str
    """Name of the COT evaluation config"""
    cot_chain: str
    """Name of the COTchain to use, must be registered in chain_registry.py"""
    description: Optional[str]
    """Description of the COT evaluation config"""
    model: str
    """HF Repo with model weights and config"""
    revision: Optional[str]
    """Revision of the model"""
    dtype: Optional[str]
    """Precision of the model"""
    modelkwargs: Optional[dict]
    """model kwargs to passed to model init function"""
    tasks: list
    """Tasks to evaluate on"""

    @classmethod
    def from_yaml(cls, path: str) -> "COTEvalConfig":
        """Load COT config from YAML file

        Args:
            path (str): Path to YAML file

        Returns:
            COTEvalConfig: COTEval config
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def to_yaml(self) -> str:
        """Dump COT config to YAML string

        Returns:
            str: YAML string
        """
        return yaml.dump(self.model_dump())

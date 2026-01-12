from collections import defaultdict
from typing import Any, Dict, Union, List
from jinja2 import Template

from mirage.variables.environment import VariableEnvironment
import logging

logger = logging.getLogger(__name__)

class TemplateRenderer():
    def __init__(self, output_schema: Dict[str, Any]]) -> None:
        self.output_schema = output_schema

    def batch_render(self, batch: List[VariableEnvironment]) -> Dict[str, List[Any]]:
        rendered_batch = defaultdict(list)
        for env in batch:
            for key, template_obj in self.output_schema:
                rendered_batch[key].append(self._fill_template_recursive(template_obj, env))

        return rendered_batch

    def _fill_template_recursive(
            self,
            template_obj: Union[Dict, List, str],
            context: VariableEnvironment) -> Any:
        """Recursively fill templates in nested structures."""
        # TODO: Fix issue when non string variables are passed in the context!
        if isinstance(template_obj, dict):
            return {
                k: self._fill_template_recursive(v, context)
                for k, v in template_obj.items()
            }
        elif isinstance(template_obj, list):
            return [self._fill_template_recursive(v, context) for v in template_obj]
        elif isinstance(template_obj, str):
            return Template(template_obj).render(context)
        else:
            logger.warning(f"Unable to render template {template_obj}")
            return template_obj


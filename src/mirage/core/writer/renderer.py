from collections import defaultdict
from typing import Any, Dict, Optional, List
from jinja2 import Template, Environment
from jinja2.nodes import Output, Name

from mirage.core.process.variables import VariableEnvironment
import logging

logger = logging.getLogger(__name__)

env = Environment()

class TemplateRenderer():
    def __init__(self, output_schema: Dict[str, Any]) -> None:
        self.output_schema = output_schema

    def batch_render(self, batch: List[VariableEnvironment]) -> Dict[str, List[Any]]:
        rendered_batch = defaultdict(list)
        for env in batch:
            for key, template_obj in self.output_schema.items():
                rendered_batch[key].append(self._fill_template_recursive(template_obj, env))

        return rendered_batch

    def is_single_variable_template(self, s: str) -> Optional[str]:
        """
        If s is exactly '{{ var }}', return 'var'.
        Otherwise return None.
        """
        ast = env.parse(s)
        
        if len(ast.body) != 1:
            return None
        
        node = ast.body[0]
        
        if not isinstance(node, Output) or len(node.nodes) != 1:
            return None

        expr = node.nodes[0]

        if isinstance(expr, Name):
            return expr.name

        return None

    def _fill_template_recursive(self, template_obj: Any, context: VariableEnvironment) -> Any:
        if isinstance(template_obj, dict):
            return {
                k: self._fill_template_recursive(v, context)
                for k, v in template_obj.items()
            }

        elif isinstance(template_obj, list):
            return [self._fill_template_recursive(v, context) for v in template_obj]

        elif isinstance(template_obj, str):
            var_name = self.is_single_variable_template(template_obj)
            context_dict = context.to_dict()

            if var_name is not None and var_name in context_dict:
                # Return the actual object, not its string representation
                return context_dict[var_name]

            # Fallback: normal Jinja rendering
            return Template(template_obj).render(context_dict)

        else:
            return template_obj




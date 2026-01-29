"""Template renderer for generating output from variable environments."""

from collections import defaultdict
from typing import Any, Dict, Optional, List
from jinja2 import Template, Environment
from jinja2.nodes import Output, Name

from mirage.core.process.variables import VariableEnvironment
import logging

logger = logging.getLogger(__name__)

JINJA_ENV = Environment()


class TemplateRenderer:
    """Renderer for generating output from variable environments using Jinja2 templates.

    Supports nested templates (dicts and lists) and optimized handling of
    simple variable references.

    Attributes:
        output_schema: Dictionary defining the structure of output samples.
    """

    def __init__(self, output_schema: Dict[str, Any]) -> None:
        """Initialize the template renderer.

        Args:
            output_schema: Dictionary defining the structure of output samples.
                Values can be strings (Jinja2 templates), lists, or dicts.
        """
        self.output_schema = output_schema

    def batch_render(self, batch: List[VariableEnvironment]) -> Dict[str, List[Any]]:
        """Render a batch of variable environments.

        Args:
            batch: List of variable environments to render.

        Returns:
            Dictionary mapping output keys to lists of rendered values.
        """
        rendered_batch = defaultdict(list)
        for env in batch:
            for key, template_obj in self.output_schema.items():
                rendered_batch[key].append(
                    self._fill_template_recursive(template_obj, env)
                )

        return rendered_batch

    def is_single_variable_template(self, s: str) -> Optional[str]:
        """Check if a string is a simple single-variable template.

        Args:
            s: String to check.

        Returns:
            The variable name if s is exactly '{{ var }}', otherwise None.
        """
        ast = JINJA_ENV.parse(s)

        if len(ast.body) != 1:
            return None

        node = ast.body[0]

        if not isinstance(node, Output) or len(node.nodes) != 1:
            return None

        expr = node.nodes[0]

        if isinstance(expr, Name):
            return expr.name

        return None

    def _fill_template_recursive(
        self, template_obj: Any, context: VariableEnvironment
    ) -> Any:
        """Recursively fill a template object with values from a variable environment.

        Args:
            template_obj: Template object (str, dict, list, or other type).
            context: Variable environment containing variable values.

        Returns:
            The rendered object with templates filled with variable values.
        """
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
                return context_dict[var_name]

            return Template(template_obj).render(context_dict)

        else:
            return template_obj

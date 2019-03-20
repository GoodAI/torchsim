import logging
import os
from typing import List, Dict, Any, Type

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval2.experiment_runner_params import ExperimentParams
from torchsim.core.eval.series_plotter import to_safe_name
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase
from torchsim.core.eval2.measurement_manager import MeasurementManager
from torchsim.core.eval2.parameter_extractor import ParameterExtractor

logger = logging.getLogger(__name__)


class DocumentPublisher:
    """A class responsible for the publishing of results based on a template."""

    _header_table_width = 60

    def __init__(self, template: ExperimentTemplateBase, docs_folder: str, template_class: Type,
                 default_topology_parameters: Dict[str, Any], topology_params: List[Dict[str, Any]],
                 runner_params: ExperimentParams):
        """Prepares the document for writing the results"""
        self._document = Document()

        self._template = template
        self._docs_folder = docs_folder
        self._template_class = template_class
        self._default_topology_parameters = default_topology_parameters
        self._topology_params = topology_params
        self._runner_params = runner_params

        parameter_extractor = ParameterExtractor(self._default_topology_parameters)
        self._common_parameters, self._differing_parameters = parameter_extractor.extract(self._topology_params)

        if not self._docs_folder:
            raise RuntimeError("Docs folder not defined")

    def publish_results(self, timestamp: str, measurement_manager: MeasurementManager):
        """Publishes the result and returns the path of the main html document."""
        self._document.add(self._get_heading(timestamp))

        self._template.publish_results(self._document, self._docs_folder, measurement_manager,
                                       self.parameters_to_string(self._differing_parameters))

        logger.info(f"Results documents generated in {self._docs_folder}")
        return self._write_document(self._docs_folder)

    def _write_document(self, docs_folder: str) -> str:
        """Writes the document to the disk."""
        if not os.path.isdir(docs_folder):
            os.makedirs(docs_folder)

        doc_path = os.path.join(docs_folder, to_safe_name(self._template.experiment_name + ".html"))
        self._document.write_file(doc_path)

        logger.info('done')
        return doc_path

    @classmethod
    def _param_to_html(cls, name: str, value: Any, indent: bool = False, bold: bool = False) -> str:
        if hasattr(value, '_asdict'):
            value = value._asdict()

        if isinstance(value, dict):
            row = cls._param_to_html(name, '', bold=True) + ''.join(
                [cls._param_to_html(key, val, indent=True) for key, val in value.items()])
        else:
            style = ' style="padding-left: 20px"' if indent else ''
            name = f'<strong>{name}</strong>' if bold else name
            row = "<tr>" + \
                  f"<td{style}>{name}</td>" + \
                  f"<td>{cls._param_value_to_string(value)}</td>" + \
                  "</tr>"
        return row

    def _runner_configuration_to_html(self) -> str:
        """Convert the runner's configuration to the string for the html heading."""
        result = f"\n<b>Experiment and runner configuration</b>:<br> "
        result += self._get_table_header()

        # values in columns
        result += self._param_to_html("runner", self._runner_params)
        for key, value in self._template.get_additional_parameters().items():
            result += self._param_to_html(key, value)
        result += "</table>"

        return result

    def _get_table_header(self):
        header = f"<table style=\"width:{self._header_table_width}%\">" + \
                 "<tr>" + \
                 "<th style=\"text-align: left\">Param name</th>" + \
                 "<th style=\"text-align:left\">Param value</th>" + \
                 "</tr>"
        return header

    @classmethod
    def parameters_to_string(cls, parameters: List[Dict[str, Any]]) -> List[str]:
        return [", ".join(f"{param}: {cls._param_value_to_string(value)}"
                          for param, value in parameter.items())
                for parameter in parameters]

    @classmethod
    def _param_value_to_string(cls, value):
        """Lists of normal values are parsed OK, param value can be also list of classes, parse to readable string."""
        if type(value) in (list, tuple):
            return [cls._param_value_to_string(x) for x in value]
        elif isinstance(value, type):
            return value.__name__
        return value

    def _get_heading(self, date: str):
        """Get heading of the html file with the experiment description"""

        info = f"<p><b>Template</b>: {self._template_class.__name__}<br>" + \
               f"\n<b>Experiment_name</b>: {self._template.experiment_name}<br>" + \
               f"\n<b>Date:</b> {date[1:]}</p>"

        info += '<table style="width: 100%">'
        info += '<tr>'
        info += '<td style="width: 50%">'
        info += f"\n<b>List of common parameters</b>:<br> "
        # create table with the params
        info += self._get_table_header()
        for key, value in self._common_parameters.items():
            info += self._param_to_html(key, value)
        info += "</table>"

        info += '</td>'
        info += '<td>'
        # add the description of the template configuration
        info += self._runner_configuration_to_html()
        info += '</td>'
        info += '</tr>'
        info += '</table>'

        return info

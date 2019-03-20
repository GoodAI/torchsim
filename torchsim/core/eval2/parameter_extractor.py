from typing import Dict, Any, List, Tuple

from torchsim.utils.dict_utils import dict_with_defaults, get_dict_intersection, remove_from_dict, to_nested_dict, \
    NestedDictException


class ParameterExtractor:
    """This class is used to extract the header and legend parameters for the DocumentPublisher."""
    def __init__(self, default_parameters: Dict[str, Any]):
        self._default_parameters = default_parameters

    def extract(self, parameters: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Extracts the header and legend parameters.

        The header parameters are those that are the same in all individual runs.
        The legend parameters are those that are different for at least some of the runs.

        Returns:
            header parameters, legend parameters
        """

        default_parameters = to_nested_dict(self._default_parameters)

        # Create the legend - add default values where they are missing.
        legend = []
        for single_parameters in parameters:
            single_parameters = to_nested_dict(single_parameters)
            try:
                defaults = dict_with_defaults(single_parameters, default_parameters)
            except NestedDictException:
                defaults = {}

            legend.append(defaults)

        # Get the intersection of all the parameter dicts in the list.
        common_parameters = legend[0]
        for dict2 in legend[1:]:
            common_parameters = get_dict_intersection(common_parameters, dict2)

        header = common_parameters

        # Remove the common parameters from each legend item - they will be displayed in the header.
        legend = [remove_from_dict(single_params, common_parameters) for single_params in legend]

        return header, legend

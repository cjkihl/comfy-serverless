import logging
import sys
import traceback
from comfy.model_management import InterruptProcessingException
from execution import format_value, full_type_name, get_input_data, get_output_data
from nodes import NODE_CLASS_MAPPINGS


def cd_recursive_execute_sync(
    prompt,
    outputs,
    current_item,
    extra_data,
    executed,
    prompt_id,
    outputs_ui,
    object_storage,
):
    unique_id = current_item
    inputs = prompt[unique_id]["inputs"]
    class_type = prompt[unique_id]["class_type"]
    class_def = NODE_CLASS_MAPPINGS[class_type]
    if unique_id in outputs:
        return

    for x in inputs:
        input_data = inputs[x]

        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            if input_unique_id not in outputs:
                for result in cd_recursive_execute_sync(
                    prompt,
                    outputs,
                    input_unique_id,
                    extra_data,
                    executed,
                    prompt_id,
                    outputs_ui,
                    object_storage,
                ):
                    yield result

    input_data_all = None
    try:
        input_data_all = get_input_data(
            inputs, class_def, unique_id, outputs, prompt, extra_data
        )
        yield {"status": "loading", "type": class_type, "node_id": unique_id}

        obj = object_storage.get((unique_id, class_type), None)
        if obj is None:
            obj = class_def()
            object_storage[(unique_id, class_type)] = obj

        output_data, output_ui = get_output_data(obj, input_data_all)
        outputs[unique_id] = output_data
        if len(output_ui) > 0:
            outputs_ui[unique_id] = output_ui
            yield {
                "status": "success",
                "type": class_type,
                "node_id": unique_id,
                "output": output_ui,
            }

    except InterruptProcessingException as iex:
        logging.info("Processing interrupted")
        # skip formatting inputs/outputs
        error_details = {
            "node_id": unique_id,
        }
        raise InterruptProcessingException(error_details) from iex
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        output_data_formatted = {}
        for node_id, node_outputs in outputs.items():
            output_data_formatted[node_id] = [
                [format_value(x) for x in l] for l in node_outputs
            ]

        logging.error("!!! Exception during processing !!!")
        logging.error(traceback.format_exc())

        error_details = {
            "node_id": unique_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted,
            "current_outputs": output_data_formatted,
        }
        raise Exception(error_details) from ex
    executed.add(unique_id)
    return

import json
import logging
import sys
import traceback
import comfy.model_management
from execution import format_value, full_type_name, get_input_data, get_output_data
import nodes

def recursive_execute(
    callback,
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
    input_data_all = None

    try:
        inputs = prompt[unique_id]["inputs"]
        class_type = prompt[unique_id]["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS.get(class_type)
        
        # Check that the current class exists in NODE_CLASS_MAPPINGS
        if class_def is None:
            raise Exception(f"The class '{class_type}' does not exist in NODE_CLASS_MAPPINGS.")
        
        if unique_id in outputs:
            return (True, None, None)

        for x in inputs:
            input_data = inputs[x]

            if isinstance(input_data, list):
                input_unique_id = input_data[0]
                if input_unique_id not in outputs:
                    result = recursive_execute(
                        callback,
                        prompt,
                        outputs,
                        input_unique_id,
                        extra_data,
                        executed,
                        prompt_id,
                        outputs_ui,
                        object_storage,
                    )
                    if result[0] is not True:
                        # Another node failed further upstream
                        return result

   
        input_data_all = get_input_data(
            inputs, class_def, unique_id, outputs, prompt, extra_data
        )
        callback({"status": "loading", "type": class_type, "node_id": unique_id})

        obj = object_storage.get((unique_id, class_type), None)
        if obj is None:
            obj = class_def()
            object_storage[(unique_id, class_type)] = obj

        output_data, output_ui = get_output_data(obj, input_data_all)
        outputs[unique_id] = output_data
        if len(output_ui) > 0:
            outputs_ui[unique_id] = output_ui
            callback(
                {
                    "status": "success",
                    "type": class_type,
                    "node_id": unique_id,
                    "output": output_ui,
                },
            )
    except comfy.model_management.InterruptProcessingException as iex:
        logging.info("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": unique_id,
        }

        return (False, error_details, iex)
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

        print("!!! Exception during processing")
        print(traceback.format_exc())

        error_details = {
            "node_id": unique_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted,
            "current_outputs": output_data_formatted,
        }
        callback({"status": "error", "error": json.dumps(error_details) })
        return (False, error_details, ex)

    executed.add(unique_id)

    return (True, None, None)

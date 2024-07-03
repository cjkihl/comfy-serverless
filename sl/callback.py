CALLBACK_HOOK = None

def set_callback_hook(function):
    global CALLBACK_HOOK
    CALLBACK_HOOK = function

def send_callback(args:object):
    global CALLBACK_HOOK
    if(CALLBACK_HOOK is not None):
        CALLBACK_HOOK(args)
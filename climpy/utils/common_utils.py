def regularize_display_in_terminal():
    '''
    display is not available in terminal. Replace it with pretty print pprint
    :return:
    '''

    import builtins

    try:
        from IPython.display import display
        builtins.display = display
    except ImportError:
        print('No display available. Switching to pretty print')
        from pprint import pprint
        builtins.display = pprint
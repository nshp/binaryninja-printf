from binaryninja import *

META_KEY_FUNCTIONS  = 'local_printf_functions'
META_KEY_EXTENSIONS = 'local_printf_extensions'
default_functions = {
    'int printf(const char *fmt, ...)'   : (0, 1),
    'int wprintf(const wchar16 *fmt, ...)'   : (0, 1),
    'int fprintf(void *stream, const char *fmt, ...)': (1, 2),
    'int dprintf(int fd, const char *fmt, ...)': (1, 2),
    'int sprintf(char *buf, const char *fmt, ...)'  : (1, 2),
    'int asprintf(char **strp, const char *fmt, ...)'  : (1, 2),
    'int snprintf(char *buf, size_t size, const char *fmt, ...)': (2, 3),
    'int __printf_chk(int flag, const char *fmt, ...)': (1, 2),
    'int __fprintf_chk(void *stream, int flag, const char *fmt, ...)': (2, 3),
    'int __sprintf_chk(char *buf, int flag, size_t buflen, const char *fmt, ...)': (3, 4),
    'int __snprintf_chk(char *buf, size_t buflen, int flag, size_t size, const char *fmt, ...)': (4, 5),
}

MAX_STRING_LENGTH = 2048

def find_expr(il, ops):
    if il.operation in ops:
        return il
    for operand in il.operands:
        oper = getattr(operand, 'operation', None)
        if oper in ops:
            return operand
        elif oper is None:
            continue
        rec = find_expr(operand, ops)
        if rec is not None:
            return rec
    return None

STATE_NOTHING = 0
STATE_FMT_1   = 1
STATE_FMT_2   = 2
STATE_FMT_3   = 3
STATE_PREC    = 4
STATE_END     = 5

FMT_SKIP = set(b'0123456789#<>')
FMT_FLAG = set(b'#0 -+\'I')
FMT_MOD  = set(b'lhzqLjt')
FMT_SPEC = {
    ord('d'): 'int',
    ord('i'): 'int',
    ord('o'): 'unsigned int',
    ord('u'): 'unsigned int',
    ord('x'): 'unsigned int',
    ord('X'): 'unsigned int',
    ord('e'): 'double',
    ord('E'): 'double',
    ord('g'): 'double',
    ord('G'): 'double',
    ord('a'): 'double',
    ord('A'): 'double',
    ord('f'): 'float',
    ord('F'): 'float',
    ord('c'): 'char',
    ord('a'): 'const char *',
    ord('s'): 'const char *',
    ord('p'): 'const void *',
    ord('n'): 'int *',
    ord('m'): '', # implicitly uses errno
}

def decide_type(ext_specs, mod, spec):
    if spec is None:
        if mod:
            spec = mod[-1]
            mod = mod[:-1]
        else:
            return None

    if chr(spec) in ext_specs:
        specs = ext_specs[chr(spec)]
        try:
            mod_str = mod.decode('utf-8')
            if mod_str in specs:
                return specs[mod_str]
        except UnicodeDecodeError:
            pass

    base_type = FMT_SPEC.get(spec, None)
    if base_type is None:
        return None

    if base_type == '':
        return ''

    if not mod:
        return base_type

    if mod == b'hh' and base_type == 'int':
        return 'char'
    elif mod == b'h' and base_type == 'int':
        return 'short'
    elif mod == b'l' and base_type == 'int':
        return 'long'
    elif mod in {b'll',b'q'} and base_type == 'int':
        return 'long long'

    if mod == b'hh' and base_type == 'int *':
        return 'char *'
    elif mod == b'h' and base_type == 'int *':
        return 'short *'
    elif mod == b'l' and base_type == 'int *':
        return 'long *'
    elif mod in {b'll',b'q'} and base_type == 'int *':
        return 'long long *'

    elif mod == b'hh' and base_type == 'unsigned int':
        return 'unsigned char'
    elif mod == b'h' and base_type == 'unsigned int':
        return 'unsigned short'
    elif mod == b'l' and base_type == 'unsigned int':
        return 'unsigned long'
    elif mod in {b'll',b'q'} and base_type == 'unsigned int':
        return 'unsigned long long'

    elif mod == b'z' and base_type == 'int':
        return 'ssize_t'
    elif mod == b'z' and base_type == 'unsigned int':
        return 'size_t'

    elif mod == b't' and base_type == 'int':
        return 'ptrdiff_t'
    elif mod == b't' and base_type == 'int *':
        return 'ptrdiff_t *'

    elif mod == b'l' and base_type == 'char':
        return 'wchar_t'
    elif mod == b'l' and base_type == 'const char *':
        return 'const wchar_t *'

    elif mod == b'j' and base_type == 'int':
        return 'ssize_t'
    elif mod == b'j' and base_type == 'unsigned int':
        return 'size_t'
    elif mod == b'j' and base_type == 'int *':
        return 'ssize_t *'

    else:
        return None

def format_types(ext_specs, fmt):
    types = []
    state = STATE_NOTHING
    has_var_prec = False
    has_var_width = False
    spec = None
    mod = []

    is_spec = lambda c: c in FMT_SPEC or chr(c) in ext_specs

    for i, c in enumerate(fmt):
        # print('state={}, c={:x}'.format(state,c))
        if state == STATE_NOTHING:
            if c == ord('%'):
                state = STATE_FMT_1
        elif state == STATE_FMT_1:
            if c == ord('%'):
                state = STATE_NOTHING
            elif c in FMT_FLAG or c in FMT_SKIP:
                state = STATE_FMT_2
            elif c == ord('.'):
                state = STATE_PREC
            elif c == ord('*'):
                has_var_width = True
                state = STATE_FMT_2
            elif c in FMT_MOD:
                mod.append(c)
                state = STATE_FMT_3
            elif is_spec(c):
                spec = c
                state = STATE_END
            else:
                return None # Invalid
        elif state == STATE_FMT_2:
            if c in FMT_SKIP:
                continue
            elif c == ord('.'):
                state = STATE_PREC
            elif c in FMT_MOD:
                mod.append(c)
                state = STATE_FMT_3
            elif is_spec(c):
                spec = c
                state = STATE_END
            else:
                return None # Invalid
        elif state == STATE_FMT_3:
            if c in FMT_MOD:
                mod.append(c)
            elif is_spec(c):
                spec = c
                state = STATE_END
            elif mod and is_spec(mod[-1]):
                spec = mod[-1]
                mod = mod[:-1]
                state = STATE_END
            else:
                return None # Invalid
        elif state == STATE_PREC:
            if c == ord('*'):
                has_var_prec = True
                state = STATE_FMT_2
            else:
                state = STATE_FMT_2
        elif state == STATE_END:
            ty = decide_type(ext_specs, bytes(mod), spec)
            if ty is None:
                return None # Invalid
            elif ty != '': # Empty string means no argument consumed, e.g. %m
                if has_var_width:
                    types.append('int')
                if has_var_prec:
                    types.append('int')
                types.append(ty)

            has_var_width = False
            has_var_prec = False
            spec = None
            mod = []
            state = STATE_NOTHING

            if c == ord('%'):
                state = STATE_FMT_1

    if state == STATE_END:
        ty = decide_type(ext_specs, bytes(mod), spec)
        if ty is None:
            return None # Invalid
        if has_var_width:
            types.append('int')
        if has_var_prec:
            types.append('int')
        types.append(ty)
    elif state != STATE_NOTHING:
        return None # String ended mid-format, invalid

    return types

# print(format_types({}, b'%-9s%lu'))
# print(format_types({}, b'%llu'))
# print(format_types({'x': {'ll': 'void *'}}, b'%llx'))
# print(format_types({}, b'%s%m%u'))
# import sys
# sys.exit(0)

def define_cstring(bv, address):
    data = bv.read(address, MAX_STRING_LENGTH)
    nul = data.find(b'\0')
    if nul == -1:
        log_warn("{:#x}: Not a string, or string too long".format(address))
        return None

    bv.define_data_var(address, Type.array(Type.char(), nul + 1))
    return data[:nul]

# Add TAILCALL ops here once "override call type" works on them
LLIL_CALLS = {LowLevelILOperation.LLIL_CALL,
              LowLevelILOperation.LLIL_CALL_STACK_ADJUST}

MLIL_CALLS = {MediumLevelILOperation.MLIL_CALL,
              MediumLevelILOperation.MLIL_TAILCALL}

class PrintfTyperBase:
    def __init__(self, view):
        self.view = view
        try:
            self.local_extns = view.query_metadata(META_KEY_EXTENSIONS)
        except KeyError:
            self.local_extns = {}

    def handle_function(self, symbol, variadic_type, fmt_pos, arg_pos, thread=None):
        bv = self.view
        # Using code refs instead of callers here to handle calls through named
        # function pointers
        calls = list(bv.get_code_refs(symbol.address))
        ncalls = len(calls)
        it = 1

        for ref in calls:
            if thread is not None:
                if thread.cancelled:
                    log_info("printf typing cancelled")
                    break
                thread.progress = "processing: {} ({}/{})".format(symbol.name, it, ncalls)
                it += 1

            mlil = ref.mlil
            mlil_index = None
            if mlil is None:
                # If there is no mlil at this address, we'll look at the LLIL
                # and scan forward until we see a call that seems to match up
                llil_instr = ref.llil
                llil = ref.function.llil
                if llil_instr is None:
                    log_info(f"no llil for {ref.address:#x}")
                    continue
                for idx in range(llil_instr.instr_index, len(llil)):
                    if llil[idx].operation in LLIL_CALLS and llil[idx].dest.value.value == symbol.address:
                        call_address = llil[idx].address
                        mlil_index = ref.function.mlil.get_instruction_start(call_address)
                        break
                    if idx > llil_instr.instr_index + 128:
                        # Don't scan forward forever...
                        break
            else:
                call_address = ref.address
                mlil_index = mlil.instr_index

            func = ref.function
            mlil = func.medium_level_il
            if mlil_index is None:
                log_info(f"no mlil index for {ref.address:#x}")
                continue

            il = mlil[mlil_index]
            call_expr = find_expr(il, MLIL_CALLS)
            if call_expr is None:
                log_debug("Cannot find call expr for ref {:#x}".format(call_address))
                continue

            if call_expr.dest.constant != symbol.address:
                log_warn("{:#x}: Call expression dest {!r} does not match {!r}".format(call_address, call_expr.dest, symbol))
                continue

            call_args = call_expr.operands[2]
            if len(call_args) <= fmt_pos:
                log_warn("Call at {:#x} does not respect function type".format(call_address))
                continue

            fmt_arg = call_args[fmt_pos]
            fmt_arg_value = fmt_arg.possible_values
            if fmt_arg_value.type in {RegisterValueType.ConstantPointerValue, RegisterValueType.ConstantValue}:
                fmt_ptr = fmt_arg_value.value
                fmt = define_cstring(bv, fmt_ptr)
                if fmt is None:
                    log_warn("{:#x}: Bad format string at {:#x}".format(call_address, fmt_ptr))
                    continue

                fmt_type_strs = format_types(self.local_extns, fmt)
                # print(fmt, fmt_type_strs)
                if fmt_type_strs is None:
                    log_warn("{:#x}: Failed to parse format string {!r}".format(call_address, fmt))
                    continue


            elif fmt_arg_value.type == RegisterValueType.InSetOfValues:
                fmts = set()
                for fmt_ptr in fmt_arg_value.values:
                    fmt = define_cstring(bv, fmt_ptr)
                    if fmt is None:
                        log_warn("{:#x}: Bad format string at {:#x}".format(call_address, fmt_ptr))
                        break
                    fmt_type_strs = format_types(self.local_extns, fmt)
                    if fmt_type_strs is None:
                        log_warn("{:#x}: Failed to parse format string {!r}".format(call_address, fmt))
                        fmt = None
                        break
                    fmts.update((tuple(fmt_type_strs),))

                if fmt is None:
                    continue
                elif not fmts:
                    log_warn("{:#x}: Unable to resolve format string from {!r}".format(call_address, fmt_arg))
                    continue
                elif len(fmts) > 1:
                    log_warn("{:#x}: Differing format types passed to one call: {!r}".format(call_address, fmts))
                    continue

                # print(fmt, fmt_type_strs)
                fmt_type_strs = fmts.pop()

            else:
                log_warn("{:#x}: Ooh, format bug? {!r} ({!r}) is not const".format(call_address, fmt_arg, fmt_arg_value))
                continue

            try:
                fmt_types = map(bv.parse_type_string, fmt_type_strs)
            except SyntaxError as e:
                log_error("Type parsing failed for {!r}: {}".format(fmt_type_strs, e))
                continue

            fmt_types = list(map(lambda t: t[0], fmt_types))

            explicit_type = Type.function(variadic_type.return_value,
                                          variadic_type.parameters + fmt_types,
                                          variable_arguments=False,
                                          calling_convention=variadic_type.calling_convention,
                                          stack_adjust=variadic_type.stack_adjustment or None)
            log_debug("{:#x}: format string {!r}: explicit type {!r}".format(call_address, fmt, explicit_type))
            func.set_call_type_adjustment(call_address, explicit_type)

class PrintfTyperSingle(BackgroundTaskThread):
    def __init__(self, view, symbol, variadic_type, fmt_pos, arg_pos):
        super().__init__("", True)
        self.view = view
        self.progress = ""
        self.symbol = symbol
        self.variadic_type = variadic_type
        self.fmt_pos = fmt_pos
        self.arg_pos = arg_pos

    def run(self):
        self.progress = "processing: {}".format(self.symbol.name)
        log_debug(self.symbol.name)
        PrintfTyperBase(self.view).handle_function(self.symbol, self.variadic_type, self.fmt_pos, self.arg_pos, self)

    def update_analysis_and_handle(self):
        AnalysisCompletionEvent(self.view, lambda: PrintfTyperSingle(self.view, self.symbol, self.variadic_type, self.fmt_pos, self.arg_pos).start())
        self.view.update_analysis()

class PrintfTyper(BackgroundTaskThread):
    def __init__(self, view):
        super().__init__("", True)
        self.view = view
        self.progress = ""

    def run(self):
        bv = self.view
        self.progress = "typing format functions"
        symbols = []

        printf_functions = default_functions.copy()

        try:
            local_funcs = bv.query_metadata(META_KEY_FUNCTIONS)
        except KeyError:
            local_funcs = {}

        func_names = {bv.parse_type_string(f)[1]: f for f in printf_functions.keys()}
        for ftype, args in local_funcs.items():
            name = bv.parse_type_string(ftype)[1]
            if name in func_names:
                del printf_functions[func_names[name]]
            func_names[name] = ftype
            printf_functions[ftype] = args

        for decl, positions in printf_functions.items():
            decl_type, name = bv.parse_type_string(decl)
            for symbol in bv.get_symbols_by_name(str(name)):
                # Handle PLTs and local functions
                if symbol.type == SymbolType.FunctionSymbol:
                    func = bv.get_function_at(symbol.address)
                    if func is None:
                        continue
                    func.set_user_type(decl_type)
                    symbols.append((symbol, decl_type, positions))
                # Handle GOT entries
                elif symbol.type == SymbolType.ImportAddressSymbol:
                    var = bv.get_data_var_at(symbol.address)
                    if var is None:
                        continue
                    if var.type.type_class != TypeClass.PointerTypeClass:
                        continue
                    var.type = Type.pointer(bv.arch,
                                            decl_type,
                                            const=var.type.const)
                    symbols.append((symbol, decl_type, positions))

        self.progress = ""
        bv.update_analysis_and_wait()

        typer = PrintfTyperBase(bv)
        for symbol, variadic_type, positions in symbols:
            if self.cancelled:
                break
            self.progress = "processing: {}".format(symbol.name)
            log_info(self.progress)
            typer.handle_function(symbol, variadic_type, positions[0], positions[1], self)

        self.progress = ""

class ExtensionDialog(object):
    from binaryninja.interaction import get_form_input, TextLineField

    def __init__(self):
        self.title = "Add custom format spec"
        self.fields = [
            TextLineField("Format spec character"),
            TextLineField("Format spec modifier (optional)"),
            TextLineField("Paremeter type"),
        ]

    def show(self):
        inp = get_form_input(self.fields, self.title)
        if not inp:
            return None
        return (self.fields[0].result, self.fields[1].result, self.fields[2].result)

def extend(bv):
    from binaryninja.interaction import show_message_box

    result = ExtensionDialog().show()
    if result is None:
        return
    spec, mod, ty_str = result

    if len(spec) != 1:
        show_message_box("Error", "Format spec must be a single character.")
        return
    try:
        bv.parse_type_string(ty_str)
    except SyntaxError as e:
        show_message_box("Error", e.msg)
        return

    try:
        local_extns = bv.query_metadata(META_KEY_EXTENSIONS)
    except KeyError:
        local_extns = {}

    if spec not in local_extns:
        local_extns[spec] = {}

    local_extns[spec][mod] = ty_str
    bv.store_metadata(META_KEY_EXTENSIONS, local_extns)

def work(bv):
    worker = PrintfTyper(bv)
    worker.start()
    bv.update_analysis()

class ArgumentSelector(object):
    from binaryninja.interaction import get_form_input, IntegerField

    def __init__(self, func):
        self.title = ("Describe format function {!r} at {:#x}"
                      .format(func.name, func.start))
        self.fields = [
            IntegerField("Format string argument index (zero-based)"),
            IntegerField("First format argument index (zero-based)"),
        ]

    def show(self):
        inp = get_form_input(self.fields, self.title)
        if not inp:
            return None
        return (self.fields[0].result, self.fields[1].result)

def work_func(bv, func):
    from binaryninja.interaction import show_message_box

    arg_positions = ArgumentSelector(func).show()
    if arg_positions is None:
        return
    fmt_arg_pos, var_arg_pos = arg_positions

    current_type = func.function_type
    if fmt_arg_pos >= len(current_type.parameters):
        show_message_box("Error",
                         (("There is no argument at index {}. You may need to"
                          +" adjust the function type to include the format"
                          +" string argument.").format(fmt_arg_pos)),
                         icon=MessageBoxIcon.ErrorIcon)
        return
    if var_arg_pos > len(current_type.parameters):
        show_message_box("Error",
                         (("There is no argument at index {} or {}-1. You may"
                          +" need to adjust the function type to include the"
                          +" arguments leading up to the first variable"
                          +" parameter.").format(fmt_arg_pos, fmt_arg_pos)),
                         icon=MessageBoxIcon.ErrorIcon)
        return


    cc = TypeBuilder.char()
    cc.const = True
    fmt_type = Type.pointer(func.arch, cc)
    fmt_arg = FunctionParameter(fmt_type, 'fmt')

    if (str(current_type.parameters[fmt_arg_pos]) != str(fmt_arg)
        or len(current_type.parameters) != var_arg_pos
        or not current_type.has_variable_arguments):
        # Need to adjust the type

        arg_types = (current_type.parameters[:fmt_arg_pos]
                    + [fmt_arg]
                    + current_type.parameters[fmt_arg_pos+1:var_arg_pos])
        new_type = Type.function(current_type.return_value,
                                 arg_types,
                                 variable_arguments=True,
                                 calling_convention=current_type.calling_convention,
                                 stack_adjust=current_type.stack_adjustment or None)

        res = show_message_box("Confirm",
                               ("New type for function {!r} at {:#x} will be: {}"
                                .format(func.name, func.start, str(new_type))),
                               buttons=MessageBoxButtonSet.YesNoCancelButtonSet,
                               icon=MessageBoxIcon.QuestionIcon)
        if res != MessageBoxButtonResult.OKButton:
            return

        func.function_type = new_type

    try:
        local_funcs = bv.query_metadata(META_KEY_FUNCTIONS)
    except KeyError:
        local_funcs = {}
        pass

    named_type = (str(func.function_type.return_value)
                  + ' '
                  + func.name
                  + '('
                  + (', '.join(map(str, func.function_type.parameters)))
                  + ', ...)')
    assert str(bv.parse_type_string(named_type)[0]) == str(func.function_type)
    local_funcs.update({named_type: (fmt_arg_pos, var_arg_pos)})
    bv.store_metadata(META_KEY_FUNCTIONS, local_funcs)

    worker = PrintfTyperSingle(bv, func.symbol, func.function_type, fmt_arg_pos, var_arg_pos)
    worker.update_analysis_and_handle()

PluginCommand.register(
    "Override printf call types",
    "Properly types printf-family calls by parsing format strings",
    work
)

PluginCommand.register_for_function(
    "Add printf-like function",
    "Mark a printf-like function for type analysis",
    work_func
)

PluginCommand.register(
    "Add printf extension",
    "Add a custom format string spec",
    extend
)

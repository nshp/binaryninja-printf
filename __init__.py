from binaryninja import *
from binaryninja.log import Logger
import itertools
import re
import sys
import traceback
from typing import Optional

# until Python 3.11
def exception_add_note(exc, /, note, *args, fallback=None, **kwargs):
    if hasattr(exc, 'add_note'):
        return exc.add_note(note, *args, **kwargs)
    if fallback is not None:
        return fallback("note for exception {}: {}".format(str(exc), note))

def get_logger(session_id: Optional[int]):
    if session_id is None:
        session_id = 0
    return Logger(session_id, "Format String Analysis")

def except_log_error(raise_exc=None, get_logger_from_args=None, /):
    if raise_exc is None:
        raise_exc = False
    if get_logger_from_args is None:
        get_logger_from_args = lambda bv, *args, **kwargs: get_logger(bv.file.session_id)
    def decorator(func, /):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                exc_type, exc, exc_tb = sys.exc_info()
                for _ in range(1):
                    if exc_tb is None:
                        break
                    exc_tb = exc_tb.tb_next
                exc.with_traceback(exc_tb)
                try:
                    logger = get_logger_from_args(*args, **kwargs)
                except:
                    logger = get_logger(None)
                    logger.log_error(traceback.format_exc())
                try:
                    logger.log_error("".join(traceback.format_exception(exc)))
                except:
                    pass
                if raise_exc:
                    raise exc
        return wrapper
    return decorator

META_KEY_FUNCTIONS  = 'local_printf_functions'
META_KEY_EXTENSIONS = 'local_printf_extensions'
default_functions = [
    {'decl': 'int           printf    (                                                                         const char *format, ...)', 'args': (0, 1), 'va_name':          'vprintf'     },
    {'decl': 'int           printf_s  (                                                                         const char *format, ...)', 'args': (0, 1), 'va_name':          'vprintf_s'   },
    {'decl': 'int         __printf_chk(                               int flag,                                 const char *format, ...)', 'args': (1, 2), 'va_name':        '__vprintf_chk' },
    {'decl': 'int          dprintf    (int fd,                                                                  const char *format, ...)', 'args': (1, 2), 'va_name':          'vdprintf'    },
    {'decl': 'int        __dprintf_chk(int fd,                        int flag,                                 const char *format, ...)', 'args': (2, 3), 'va_name':        '__vdprintf_chk'},
    {'decl': 'int          fprintf    (FILE *stream,                                                            const char *format, ...)', 'args': (1, 2), 'va_name':          'vfprintf'    },
    {'decl': 'int          fprintf_s  (FILE *stream,                                                            const char *format, ...)', 'args': (1, 2), 'va_name':          'vfprintf_s'  },
    {'decl': 'int        __fprintf_chk(FILE *stream,                  int flag,                                 const char *format, ...)', 'args': (2, 3), 'va_name':        '__vfprintf_chk'},
    {'decl': 'int          sprintf    (char *buf,                                                               const char *format, ...)', 'args': (1, 2), 'va_name':          'vsprintf'    },
    {'decl': 'int          sprintf_s  (char *buf,                              rsize_t buf_len,                 const char *format, ...)', 'args': (2, 3), 'va_name':          'vsprintf_s'  },
    {'decl': 'int        __sprintf_chk(char *buf,                     int flag, size_t buf_len,                 const char *format, ...)', 'args': (3, 4), 'va_name':        '__vsprintf_chk'},
    {'decl': 'int         snprintf    (char *buf,                               size_t buf_len,                 const char *format, ...)', 'args': (2, 3), 'va_name':         'vsnprintf'    },
    {'decl': 'int         snprintf_s  (char *buf,                              rsize_t buf_len,                 const char *format, ...)', 'args': (2, 3), 'va_name':         'vsnprintf_s'  },
    {'decl': 'int        _snprintf_s  (char *buf,                               size_t buf_len, size_t max_len, const char *format, ...)', 'args': (3, 4), 'va_name':        '_vsnprintf_s'  },
    {'decl': 'int       __snprintf_chk(char *buf, size_t max_buf_len, int flag, size_t buf_len,                 const char *format, ...)', 'args': (4, 5), 'va_name':       '__vsnprintf_chk'},
    {'decl': 'int         asprintf    (char **buf_ptr,                                                          const char *format, ...)', 'args': (1, 2), 'va_name':         'vasprintf'    },
    {'decl': 'int       __asprintf_chk(char **buf_ptr,                int flag,                                 const char *format, ...)', 'args': (2, 3), 'va_name':       '__vasprintf_chk'},
    {'decl': 'int   obstack_printf    (void *obstack,                                                           const char *format, ...)', 'args': (1, 2), 'va_name':   'obstack_vprintf'    },
    {'decl': 'int __obstack_printf_chk(void *obstack,                 int flag,                                 const char *format, ...)', 'args': (2, 3), 'va_name': '__obstack_vprintf_chk'},

    {'decl': 'int    wprintf    (                                                                            const wchar_t *format, ...)', 'args': (0, 1), 'va_name':    'vwprintf'    },
    {'decl': 'int    wprintf_s  (                                                                            const wchar_t *format, ...)', 'args': (0, 1), 'va_name':    'vwprintf_s'  },
    {'decl': 'int  __wprintf_chk(                                  int flag,                                 const wchar_t *format, ...)', 'args': (0, 1), 'va_name':  '__vwprintf_chk'},
    {'decl': 'int   fwprintf    (FILE *stream,                                                               const wchar_t *format, ...)', 'args': (1, 2), 'va_name':   'vfwprintf'    },
    {'decl': 'int   fwprintf_s  (FILE *stream,                                                               const wchar_t *format, ...)', 'args': (1, 2), 'va_name':   'vfwprintf_s'  },
    {'decl': 'int   swprintf    (wchar_t *buf,                               size_t buf_len,                 const wchar_t *format, ...)', 'args': (2, 3), 'va_name':   'vswprintf'    },
    {'decl': 'int __swprintf_chk(wchar_t *buf, size_t max_buf_len, int flag, size_t buf_len,                 const wchar_t *format, ...)', 'args': (2, 3), 'va_name': '__vswprintf_chk'},
    {'decl': 'int   swprintf_s  (wchar_t *buf,                              rsize_t buf_len,                 const wchar_t *format, ...)', 'args': (2, 3), 'va_name':   'vswprintf_s'  },
    {'decl': 'int _snwprintf    (wchar_t *buf,                               size_t buf_len,                 const wchar_t *format, ...)', 'args': (2, 3), 'va_name': '_vsnwprintf'    },
    {'decl': 'int  snwprintf_s  (wchar_t *buf,                              rsize_t buf_len,                 const wchar_t *format, ...)', 'args': (2, 3), 'va_name':  'vsnwprintf_s'  },
    {'decl': 'int _snwprintf_s  (wchar_t *buf,                               size_t buf_len, size_t max_len, const wchar_t *format, ...)', 'args': (3, 4), 'va_name': '_vsnwprintf_s'  },
    {'decl': 'int  aswprintf    (wchar_t **buf_ptr,                                                          const wchar_t *format, ...)', 'args': (1, 2), 'va_name':  'vaswprintf'    },

    {'decl': 'int   __stdio_common_vfprintf  (uint64_t options, FILE *stream,                                 const  char   *format, void *locale, va_list arg)', 'args': (2, 4)},
    {'decl': 'int   __stdio_common_vfprintf_s(uint64_t options, FILE *stream,                                 const  char   *format, void *locale, va_list arg)', 'args': (2, 4)},
    {'decl': 'int  __stdio_common_vfwprintf  (uint64_t options, FILE *stream,                                 const wchar_t *format, void *locale, va_list arg)', 'args': (2, 4)},
    {'decl': 'int  __stdio_common_vfwprintf_s(uint64_t options, FILE *stream,                                 const wchar_t *format, void *locale, va_list arg)', 'args': (2, 4)},
    {'decl': 'int   __stdio_common_vsprintf  (uint64_t options,  char   *buf, size_t buf_len,                 const  char   *format, void *locale, va_list arg)', 'args': (3, 5)},
    {'decl': 'int   __stdio_common_vsprintf_s(uint64_t options,  char   *buf, size_t buf_len,                 const  char   *format, void *locale, va_list arg)', 'args': (3, 5)},
    {'decl': 'int  __stdio_common_vsnprintf_s(uint64_t options,  char   *buf, size_t buf_len, size_t max_len, const  char   *format, void *locale, va_list arg)', 'args': (4, 6)},
    {'decl': 'int  __stdio_common_vswprintf  (uint64_t options, wchar_t *buf, size_t buf_len,                 const wchar_t *format, void *locale, va_list arg)', 'args': (3, 5)},
    {'decl': 'int  __stdio_common_vswprintf_s(uint64_t options, wchar_t *buf, size_t buf_len,                 const wchar_t *format, void *locale, va_list arg)', 'args': (3, 5)},
    {'decl': 'int __stdio_common_vsnwprintf_s(uint64_t options, wchar_t *buf, size_t buf_len, size_t max_len, const wchar_t *format, void *locale, va_list arg)', 'args': (4, 6)},
]

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
FMT_MOD  = set(b'hljztLq')
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

def define_cstring(bv, address, char_type = None):
    from binaryninja.enums import Endianness
    LittleEndian = Endianness.LittleEndian
    BigEndian = Endianness.BigEndian

    if char_type is None:
        char_type = Type.char()
    char_width = char_type.width
    logger = get_logger(bv.file.session_id)
    reader = bv.reader(address)
    def reading_iter():
        for i in range(MAX_STRING_LENGTH):
            c = int.from_bytes(
                reader.read(char_width),
                byteorder='little' if reader.endianness == LittleEndian else 'big',
                signed=False,
            )
            if c == 0:
                break
            yield c
        else:
            return
    try:
        data = (bytes if char_width == 1 else list)(reading_iter())
    except StopIteration as exc:
        logger.log_warn("{:#x}: Not a string, or string too long".format(address))
        return None

    bv.define_data_var(address, Type.array(char_type, len(data) + 1))
    return data

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

    def handle_function(self, symbol, func_type, fmt_arg_pos, var_arg_pos, thread=None):
        bv = self.view
        logger = get_logger(bv.file.session_id)
        char_type = func_type.parameters[fmt_arg_pos].type.target
        # Using code refs instead of callers here to handle calls through named
        # function pointers
        calls = list(bv.get_code_refs(symbol.address))
        ncalls = len(calls)
        it = 1

        for ref in calls:
            if thread is not None:
                if thread.cancelled:
                    logger.log_info("printf typing cancelled")
                    break
                thread.progress = "processing: {} ({}/{})".format(symbol.full_name, it, ncalls)
                it += 1

            mlil = ref.mlil
            mlil_index = None
            if mlil is None:
                # If there is no mlil at this address, we'll look at the LLIL
                # and scan forward until we see a call that seems to match up
                llil_instr = ref.llil
                llil = ref.function.llil
                if llil_instr is None:
                    logger.log_info(f"no llil for {ref.address:#x}")
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
                logger.log_info(f"no mlil index for {ref.address:#x}")
                continue

            il = mlil[mlil_index]
            call_expr = find_expr(il, MLIL_CALLS)
            if call_expr is None:
                logger.log_debug("Cannot find call expr for ref {:#x}".format(call_address))
                continue

            if call_expr.dest.constant != symbol.address:
                logger.log_warn("{:#x}: Call expression dest {!r} does not match {!r}".format(call_address, call_expr.dest, symbol))
                continue

            call_args = call_expr.operands[2]
            if len(call_args) <= fmt_arg_pos:
                logger.log_warn("Call at {:#x} does not respect function type".format(call_address))
                continue

            fmt_arg = call_args[fmt_arg_pos]
            fmt_arg_value = fmt_arg.possible_values
            if fmt_arg_value.type in {RegisterValueType.ConstantPointerValue, RegisterValueType.ConstantValue}:
                fmt_ptr = fmt_arg_value.value
                fmt = define_cstring(bv, fmt_ptr, char_type)
                if fmt is None:
                    logger.log_warn("{:#x}: Bad format string at {:#x}".format(call_address, fmt_ptr))
                    continue

                fmt_type_strs = format_types(self.local_extns, fmt)
                # print(fmt, fmt_type_strs)
                if fmt_type_strs is None:
                    logger.log_warn("{:#x}: Failed to parse format string {!r}".format(call_address, fmt))
                    continue


            elif fmt_arg_value.type == RegisterValueType.InSetOfValues:
                fmts = set()
                for fmt_ptr in fmt_arg_value.values:
                    fmt = define_cstring(bv, fmt_ptr, char_type)
                    if fmt is None:
                        logger.log_warn("{:#x}: Bad format string at {:#x}".format(call_address, fmt_ptr))
                        break
                    fmt_type_strs = format_types(self.local_extns, fmt)
                    if fmt_type_strs is None:
                        logger.log_warn("{:#x}: Failed to parse format string {!r}".format(call_address, fmt))
                        fmt = None
                        break
                    fmts.update((tuple(fmt_type_strs),))

                if fmt is None:
                    continue
                elif not fmts:
                    logger.log_warn("{:#x}: Unable to resolve format string from {!r}".format(call_address, fmt_arg))
                    continue
                elif len(fmts) > 1:
                    logger.log_warn("{:#x}: Differing format types passed to one call: {!r}".format(call_address, fmts))
                    continue

                # print(fmt, fmt_type_strs)
                fmt_type_strs = fmts.pop()

            else:
                logger.log_warn("{:#x}: Ooh, format bug? {!r} ({!r}) is not const".format(call_address, fmt_arg, fmt_arg_value))
                continue

            try:
                fmt_types = map(bv.parse_type_string, fmt_type_strs)
            except SyntaxError as e:
                logger.log_error("Type parsing failed for {!r}: {}".format(fmt_type_strs, e))
                continue

            fmt_types = list(map(lambda t: t[0], fmt_types))

            explicit_type = Type.function(func_type.return_value,
                                          func_type.parameters + fmt_types,
                                          variable_arguments=False,
                                          calling_convention=func_type.calling_convention,
                                          stack_adjust=func_type.stack_adjustment or None)
            logger.log_debug("{:#x}: format string {!r}: explicit type {!r}".format(call_address, fmt, explicit_type))
            func.set_call_type_adjustment(call_address, explicit_type)

class PrintfTyperSingle(BackgroundTaskThread):
    def __init__(self, view, symbol, func_type, fmt_arg_pos, var_arg_pos):
        super().__init__("", True)
        self.view = view
        self.progress = ""
        self.symbol = symbol
        self.func_type = func_type
        self.fmt_arg_pos = fmt_arg_pos
        self.var_arg_pos = var_arg_pos

    @except_log_error(False, lambda self, *args, **kwargs: get_logger(self.view.file.session_id))
    def run(self):
        logger = get_logger(self.view.file.session_id)
        self.progress = "processing: {}".format(self.symbol.full_name)
        logger.log_debug(self.symbol.full_name)
        PrintfTyperBase(self.view).handle_function(self.symbol, self.func_type, self.fmt_arg_pos, self.var_arg_pos, self)

    def update_analysis_and_handle(self):
        logger = get_logger(self.view.file.session_id)
        AnalysisCompletionEvent(self.view, lambda: PrintfTyperSingle(self.view, self.symbol, self.func_type, self.fmt_arg_pos, self.var_arg_pos).start())
        logger.log_debug("queued PrintfTyperSingle(...).start(): {}".format(self.symbol.full_name))
        self.view.update_analysis()

class PrintfTyper(BackgroundTaskThread):
    def __init__(self, view):
        super().__init__("", True)
        self.view = view
        self.progress = ""

    @except_log_error(False, lambda self, *args, **kwargs: get_logger(self.view.file.session_id))
    def run(self):
        bv = self.view
        logger = get_logger(bv.file.session_id)
        self.progress = "typing format functions"
        logger.log_info(self.progress)
        symbols = []

        try:
            local_funcs = bv.query_metadata(META_KEY_FUNCTIONS)
        except KeyError:
            local_funcs = {}

        printf_functions = {}
        for func_info in itertools.chain(default_functions, local_funcs.values()):
            func_info = func_info.copy()
            if (func_type := func_info.get('type', None)) is not None:
                func_name = func_info.pop('name')
            else:
                func_decl = func_info['decl']
                func_name = func_info.pop('name', None)
                if func_name is None:
                    func_name = ''
                func_type, func_decl_name = bv.parse_type_string(func_decl)
                if func_name == '':
                    func_name = func_decl_name
                func_info['type'] = func_type
            printf_functions[func_name] = func_info
            logger.log_debug("recording format function: {}".format(func_name))
            if (func_va_name := func_info.get('va_name', None)) is not None:
                del func_info['va_name']
                func_va_info = func_info.copy()
                func_va_decl = func_va_info['decl']
                func_va_decl = re.sub(r'\.\.\.', 'va_list arg', func_decl, count=1)
                func_va_info['decl'] = func_va_decl
                func_va_info['type'], func_va_decl_name = bv.parse_type_string(func_va_decl)
                printf_functions[func_va_name] = func_va_info
                logger.log_debug("recording va format function: {}".format(func_va_name))
        logger.log_debug("recorded format functions")

        for func_name, func_info in printf_functions.items():
            logger.log_debug("attempting to type format function: {}".format(func_name))
            func_type = func_info['type']
            arg_positions = func_info['args']
            for symbol in bv.get_symbols_by_name(str(func_name)):
                logger.log_debug("typing format function symbol: {}".format(symbol.full_name))
                # Handle PLTs and local functions
                if symbol.type == SymbolType.FunctionSymbol:
                    func = bv.get_function_at(symbol.address)
                    if func is None:
                        continue
                    func.set_user_type(func_type)
                    symbols.append((symbol, func_type, arg_positions))
                # Handle GOT entries
                elif symbol.type == SymbolType.ImportAddressSymbol:
                    var = bv.get_data_var_at(symbol.address)
                    if var is None:
                        continue
                    if var.type.type_class != TypeClass.PointerTypeClass:
                        continue
                    var.type = Type.pointer(bv.arch,
                                            func_type,
                                            const=var.type.const)
                    symbols.append((symbol, func_type, arg_positions))

        self.progress = ""
        bv.update_analysis_and_wait()

        typer = PrintfTyperBase(bv)
        for symbol, func_type, (fmt_arg_pos, var_arg_pos) in symbols:
            if self.cancelled:
                break
            self.progress = "processing: {}".format(symbol.full_name)
            logger.log_info(self.progress)
            if func_type.has_variable_arguments:
                logger.log_debug("processing type with variable arguments: {}".format(symbol.full_name))
                typer.handle_function(symbol, func_type, fmt_arg_pos, var_arg_pos, self)

        self.progress = ""
        bv.update_analysis()

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

class ArgumentSelector(object):
    from binaryninja.interaction import get_form_input, IntegerField, TextLineField

    def __init__(self, func):
        self.title = ("Describe format function {!r} at {:#x}"
                      .format(func.symbol.full_name, func.start))
        self.fields = [
            TextLineField("String character type", "char"),
            IntegerField("Format string argument index (zero-based)"),
            IntegerField("First format argument index (zero-based)"),
        ]

    def show(self):
        inp = get_form_input(self.fields, self.title)
        if not inp:
            return None
        return (self.fields[0].result, (self.fields[1].result, self.fields[2].result))

def work_func(bv, func):
    from binaryninja.interaction import show_message_box

    logger = get_logger(bv.file.session_id)

    args_res = ArgumentSelector(func).show()
    if args_res is None:
        return
    char_decl, (fmt_arg_pos, var_arg_pos) = args_res

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


    cc = bv.parse_type_string(char_decl)[0].mutable_copy()
    cc.const = True
    fmt_type = Type.pointer(func.arch, cc)
    fmt_arg = FunctionParameter(fmt_type, 'format')

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
                                .format(func.symbol.full_name, func.start, str(new_type))),
                               buttons=MessageBoxButtonSet.YesNoCancelButtonSet,
                               icon=MessageBoxIcon.QuestionIcon)
        if res != MessageBoxButtonResult.YesButton:
            logger.log_debug("message box result wasn't yes: {}".format(res))
            return

        func.function_type = new_type

    try:
        local_funcs = bv.query_metadata(META_KEY_FUNCTIONS)
    except KeyError:
        local_funcs = {}
        pass

    func_full_name = func.symbol.full_name
    if re.match('[!().:<>?@`]', func_full_name):
        func_full_name = '`' + re.sub(r'`', '\\`', func_full_name, count=0) + '`'
    func_decl = (str(func.function_type.return_value)
                 + ' '
                 + func.symbol.full_name
                 + '('
                 + (', '.join(map(str, func.function_type.parameters)))
                 + ', ...)')
    logger.log_debug("new func_decl: {}".format(func_decl))
    try:
        func_decl_type, func_decl_name = bv.parse_type_string(func_decl)
    except BaseException as exc:
        exception_add_note(exc, "type decl string: {}".format(str(func_decl)), fallback=logger.log_warn)
        raise exc
    if str(func_decl_type) != str(func.function_type):
        raise ValueError('expected parsed type of decl {} to be {}, but was {}'.format(func_decl, str(func.function_type), str(func_decl_type)))
    if str(func_decl_name) != str(func.symbol.full_name):
        raise ValueError('expected parsed full name of decl {} to be {}, but was {}'.format(func_decl, str(func.symbol.full_name), str(func_decl_name)))
    local_funcs.update({func.symbol.full_name: {'decl': func_decl, 'args': (fmt_arg_pos, var_arg_pos)}})
    bv.store_metadata(META_KEY_FUNCTIONS, local_funcs)

    worker = PrintfTyperSingle(bv, func.symbol, func.function_type, fmt_arg_pos, var_arg_pos)
    worker.update_analysis_and_handle()

PluginCommand.register(
    "Format String Analysis\\Override printf call types",
    "Properly types printf-family calls by parsing format strings",
    except_log_error()(work)
)

PluginCommand.register_for_function(
    "Format String Analysis\\Add printf-like function",
    "Mark a printf-like function for type analysis",
    except_log_error()(work_func)
)

PluginCommand.register(
    "Format String Analysis\\Add printf extension",
    "Add a custom format string spec",
    except_log_error()(extend)
)

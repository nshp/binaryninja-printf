# Binary Ninja Printf Analysis

Plugin to update the printf family of functions:

 - printf
 - wprintf
 - fprintf
 - dprintf
 - sprintf
 - asprintf
 - snprintf
 - __printf_chk
 - __fprintf_chk
 - __sprintf_chk
 - __snprintf_chk

 Can parse existing printf family of functions using the `Override printf call types` command-palette or plugin menu action. Uses the `set_call_type_adjustment` API along with a basic format string parser to appropriate type each location where one of those APIs is called.

 Additionally, supports the ability to add additional custom printf-like functions and add new format specifiers. To add new specifiers, use the `Add printf extension` menu/action and to add new functions use the `Add printf-like function` menu/action.

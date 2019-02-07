import typing

def is_instance (obj:typing.Any, T:typing.Any, *, or_raise:typing.Any=TypeError, obj_name:str='obj') -> None:
    if not isinstance(obj, T):
        raise or_raise(f'expected {obj_name} to be an instance of {T}, but its type was actually {type(obj)}')

def is_equal (a:typing.Any, b:typing.Any, *, or_raise:typing.Any=ValueError, a_name:str='a', b_name:str='b') -> None:
    if not (a == b):
        raise or_raise(f'expected {a_name} (which was {a}) to equal {b_name} (which was {b})')

import omni.usd
from pxr import UsdGeom, UsdLux, Usd, Sdf, Gf, Vt, Tf, UsdSemantics, UsdShade # Standard USD libraries

def get_semantic_class(prim):
    attr = prim.GetAttribute("semantics:labels:class")
    if attr.HasValue():
        val = attr.Get()
        # Check if it's an array (list-like) or a single token
        if hasattr(val, "__len__") and len(val) > 0:
        # Always get the last token
            return str(val[-1])
        elif val:
            return str(val)
    return None

def set_unique_attribute(prim, attribute, type, value):
    if not prim:
        return

    if not prim.HasAttribute(attribute):
        prim.CreateAttribute(attribute, type)

    attr = prim.GetAttribute(attribute)
    attr.Clear()
    attr.Set(value)

def get_attribute(prim, attribute):
    return prim.GetAttribute(attribute).Get()
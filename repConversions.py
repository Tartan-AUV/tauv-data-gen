import omni.usd
import omni.replicator.core as rep
from pxr import Sdf, UsdSemantics


# CONVERSION FUNCTIONS

def replicator_item_to_path(rep_item):
    """
    Finds the USD path for a Replicator Item by checking OmniGraph attributes.
    """
    node = rep_item.node
    
    #Check outputs
    attr_out = node.get_attribute("outputs:prims")
    if attr_out:
        val = attr_out.get()
        if val:
            if isinstance(val, (str, Sdf.Path)):
                return str(val)
            if hasattr(val, "__len__") and len(val) > 0:
                return str(val[0])

    # Check inputs
    attr_in = node.get_attribute("inputs:prims")
    if attr_in:
        val = attr_in.get()
        if val:
            if isinstance(val, (str, Sdf.Path)):
                return str(val)
            if hasattr(val, "__len__") and len(val) > 0:
                return str(val[0])     
    return None

def replicator_item_to_prim(rep_item):
    path = replicator_item_to_path(rep_item)
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    return prim

def prim_to_path(prim):
    return prim.GetPath()

def path_to_prim(path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    return prim

def path_to_replicator_item(path):
    # Assuming it is a replicator item
    return rep.get.prim_at_path(str(path)) # note: the name .get.prims is a lie, actually gets a replicatorItem

def prim_to_replicator_item(prim):
    return path_to_replicator_item(prim_to_path(prim))
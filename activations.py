# CREDIT: https://github.com/digantamisra98/Echo

import echo.functional as Func
import echo.apl as apl
import echo.aria2 as aria2
import echo.bent_id as bent_id
import echo.beta_mish as beta_mish
import echo.brelu as brelu
import echo.elish as elish
import echo.eswish as eswish
import echo.fts as fts
import echo.hard_elish as hard_elish
import echo.isrlu as isrlu
import echo.isru as isru
import echo.lecun_tanh as lecun_tanh
import echo.maxout as maxout
import echo.mila as mila
import echo.mish as mish
import echo.silu as silu
import echo.sine_relu as sine_relu
import echo.soft_clipping as soft_clipping
import echo.soft_exponential as soft_exponential
import echo.sqnl as sqnl
import echo.srelu as srelu
import echo.swish as swish
import echo.weightedTanh as weightedTanh

# def act_(act_type='mish', **kwargs):
#     if act_type=='mish': act_fn = mish.Mish()
#     elif act_type=='aria2': act_fn = aria2.Aria2(**kwargs)
#     elif act_type=='bentid': act_fn = bent_id.BentID()
#     elif act_type=='betamish': act_fn = beta_mish.BetaMish(**kwargs)
#     elif act_type=='brelu': act_fn = brelu.BReLU()
#     elif act_type=='elish': act_fn = elish.Elish()
#     elif act_type=='eswish': act_fn = eswish.Eswish(**kwargs)
#     elif act_type=='fts': act_fn = fts.FTS()
#     elif act_type=='hardelish': act_fn = hard_elish.HardElish()
#     elif act_type=='isrlu': act_fn = isrlu.ISRLU(**kwargs)
#     elif act_type=='isru': act_fn = isru.ISRU(**kwargs)
#     elif act_type=='lecuntanh': act_fn = lecun_tanh.LeCunTanh()
#     elif act_type=='maxout': act_fn = maxout.Maxout()
#     elif act_type=='mila': act_fn = mila.Mila(**kwargs)
#     elif act_type=='silu': act_fn = silu.Silu(**kwargs)
#     elif act_type=='sinerelu': act_fn = sine_relu.SineReLU(**kwargs)
#     elif act_type=='softclipping': act_fn = soft_clipping.SoftClipping(**kwargs)
#     elif act_type=='softexponential': act_fn = soft_exponential.SoftExponential(**kwargs)
#     elif act_type=='sqnl': act_fn = sqnl.SQNL()
#     elif act_type=='srelu': act_fn = srelu.SReLU(**kwargs)
#     elif act_type=='swish': act_fn = swish.Swish(**kwargs)
#     elif act_type=='weightedtanh': act_fn = weightedTanh.WeightedTanh(**kwargs)
        
#     return act_fn

def act_(act_type='mish', **kwargs):
    if act_type=='mish': return mish.Mish()
    elif act_type=='aria2': return aria2.Aria2(**kwargs)
    elif act_type=='bentid': return bent_id.BentID()
    elif act_type=='betamish': return beta_mish.BetaMish(**kwargs)
    elif act_type=='brelu': return brelu.BReLU()
    elif act_type=='elish': return elish.Elish()
    elif act_type=='eswish': return eswish.Eswish(**kwargs)
    elif act_type=='fts': return fts.FTS()
    elif act_type=='hardelish': return hard_elish.HardElish()
    elif act_type=='isrlu': return isrlu.ISRLU(**kwargs)
    elif act_type=='isru': return isru.ISRU(**kwargs)
    elif act_type=='lecuntanh': return lecun_tanh.LeCunTanh()
    elif act_type=='maxout': return maxout.Maxout()
    elif act_type=='mila': return mila.Mila(**kwargs)
    elif act_type=='silu': return silu.Silu(**kwargs)
    elif act_type=='sinerelu': return sine_relu.SineReLU(**kwargs)
    elif act_type=='softclipping': return soft_clipping.SoftClipping(**kwargs)
    elif act_type=='softexponential': return soft_exponential.SoftExponential(**kwargs)
    elif act_type=='sqnl': return sqnl.SQNL()
    elif act_type=='srelu': return srelu.SReLU(**kwargs)
    elif act_type=='swish': return swish.Swish(**kwargs)
    elif act_type=='weightedtanh': return weightedTanh.WeightedTanh(**kwargs)

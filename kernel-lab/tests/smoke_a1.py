import sys
sys.path.insert(0,"/lab"); sys.path.insert(0,"/lab/fixtures")
import torch, time
import vllm.models.minimax_m3.common.ops.sparse_attn as sa
from vllm_exl3.sparse_attn_tiled import sparse_attn_tiled
import sparse_selection as sel
DEV="cuda"; H,KVH,D,TOPK=32,2,128,16
t0=time.time()
ti,cu,sl,pl=sel.make_selection([2048],[2048],KVH,11,DEV,"real_like",swap=0,window=64)
print("gen",time.time()-t0,"s", ti.shape, sel.union_stats(ti,cu,pl,8))
nblk=(4096+127)//128+1
kv=(torch.randn(nblk,KVH,128,2*D,device=DEV)*0.5).bfloat16()
bt=torch.arange(nblk,dtype=torch.int32,device=DEV)[None,:].repeat(8,1).contiguous()
q=(torch.randn(2048,H,D,device=DEV)*0.5).bfloat16()
ref=torch.zeros_like(q); out=torch.zeros_like(q)
sa.minimax_m3_sparse_attn(q,kv,ti,bt,cu,sl,pl,2048,KVH,D**-0.5,ref)
sparse_attn_tiled(q,kv,ti,bt,cu,sl,pl,KVH,D**-0.5,out,tile=8,num_warps=8,num_stages=1)
torch.cuda.synchronize()
d=(out.float()-ref.float()); print("rel", (d.norm()/ref.float().norm()).item(), "max", d.abs().max().item())

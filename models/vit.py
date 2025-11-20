import torch, torch.nn as nn, torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=192, patch=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x); B,C,H,W = x.shape
        return x.flatten(2).transpose(1,2), (H,W)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, p=0.0):
        super().__init__(); h=int(dim*mlp_ratio)
        self.fc1=nn.Linear(dim,h); self.act=nn.GELU(); self.fc2=nn.Linear(h,dim); self.drop=nn.Dropout(p)
    def forward(self,x): return self.drop(self.fc2(self.act(self.fc1(x))))

class Attention(nn.Module):
    def __init__(self, dim, heads=3, p=0.0):
        super().__init__(); self.h=heads; d=dim//heads; self.scale=d**-0.5
        self.qkv=nn.Linear(dim, dim*3); self.proj=nn.Linear(dim, dim); self.drop=nn.Dropout(p)
    def forward(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.h,C//self.h).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attn=(q@k.transpose(-2,-1))*self.scale; attn=attn.softmax(-1)
        x=(attn@v).transpose(1,2).reshape(B,N,C); return self.drop(self.proj(x))

class DropPath(nn.Module):
    def __init__(self,p=0.0): super().__init__(); self.p=p
    def forward(self,x):
        if self.p==0.0 or not self.training: return x
        keep=1-self.p; shape=(x.shape[0],)+(1,)*(x.ndim-1)
        return x.div(keep)*x.new_empty(shape).bernoulli_(keep)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dp=0.0):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=Attention(dim,heads); self.dp1=DropPath(dp)
        self.n2=nn.LayerNorm(dim); self.mlp=MLP(dim,mlp_ratio); self.dp2=DropPath(dp)
        # note: dropout path increases across depth in DeiT; kept simple here
    def forward(self,x): x=x+self.dp1(self.attn(self.n1(x))); x=x+self.dp2(self.mlp(self.n2(x))); return x

class DeiTTiny(nn.Module):
    def __init__(self, img_size=160, patch=16, embed_dim=192, depth=12, heads=3, mlp_ratio=4.0, drop_path=0.1):
        super().__init__()
        self.embed=PatchEmbed(3,embed_dim,patch); num_p=(img_size//patch)*(img_size//patch)
        self.cls=nn.Parameter(torch.zeros(1,1,embed_dim)); self.pos=nn.Parameter(torch.zeros(1,1+num_p,embed_dim))
        dpr=torch.linspace(0,drop_path,steps=depth).tolist()
        self.blocks=nn.ModuleList([Block(embed_dim,heads,mlp_ratio,dp=dpr[i]) for i in range(depth)])
        self.norm=nn.LayerNorm(embed_dim); self.head=nn.Linear(embed_dim,2)
        nn.init.trunc_normal_(self.pos,std=0.02); nn.init.trunc_normal_(self.cls,std=0.02)
    def forward(self,x):
        tok,(H,W)=self.embed(x); B,N,C=tok.shape; cls=self.cls.expand(B,-1,-1); pos=self.pos
        if pos.shape[1]!=N+1:
            pos_tok=pos[:,1:,:]; side=int(pos_tok.shape[1]**0.5)
            pos_grid=pos_tok.reshape(1,side,side,C).permute(0,3,1,2)
            pos_grid=F.interpolate(pos_grid,size=(H,W),mode='bicubic',align_corners=False)
            pos_tok=pos_grid.permute(0,2,3,1).reshape(1,H*W,C); pos=torch.cat([pos[:,:1,:],pos_tok],1)
        x=torch.cat([cls,tok],1)+pos[:,:N+1,:]
        for b in self.blocks: x=b(x)
        x=self.norm(x)
        cls_out=self.head(x[:,0])
        grid=x[:,1:].transpose(1,2).reshape(B,C,H,W)
        return cls_out,grid,(H,W)

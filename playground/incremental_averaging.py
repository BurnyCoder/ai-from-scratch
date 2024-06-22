import torch

# consider the following toy example:

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

wei = torch.tril(torch.ones(T, T))
wei[:, 0] = 2
wei[0, :] = 2
sum1 = wei.sum(1, keepdim=True)
sum2 = wei.sum(1, keepdim=False)
weis = wei / wei.sum(1, keepdim=True)
xbow2 = weis @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)

print(f"wei: {wei}\nweis: {weis}\nsum1: {sum1}\nsum2: {sum2}\nxbow: {xbow}\nxbow2: {xbow2}")

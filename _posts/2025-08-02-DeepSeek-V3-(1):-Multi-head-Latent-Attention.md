My interest involves investigating whether various DNN models can be executed efficiently on modern hardware accelerators. For quick reference in my daily work, this series explores the DeepSeek V3 base model, focusing on extracting computation details using the hyperparameters from the 671B variant.

Let's begin by examining how the Multi-head Latent Attention (MLA) component is computed.

<p align="center"><img src="/images/ds-v3-mla.png" width="70%"/><br>Fig.1 DeepSeek V3 MLA</p>

| No.  | Equation                                                  | Shape                                                             |
|:-----|:----------------------------------------------------------|:------------------------------------------------------------------|
| $1$  | $q = wqb @ \text{norm}(x @ wqa) $                         | $[N,L, 24576]=[24576,1536]@\text{norm}([1536,7168]@[N, L, 7168])$ |
|      | $q=\text{reshape}(q, [N,L,128,192])$                      | $[N,L,128,192] = \text{reshape}([N,L,24576])$                     |
|      | $q_{\text{nope}},q_{pe} = \text{split}(q)$                | $[N,L,128,128]\ [N,L,128,64] = \text{split}(N,L,128,192)$         |
| $2$  | $q_{pe} = \text{rotary\_emb}(q_{pe})$                     | $[N,L,128,64] = \text{rotary\_emb}([N,L,128,64])$                 |
| $3$  | $kv = wkva @ x$                                           | $[N,L,576]=[N,L,7168]@[576,7168]$                                 |
|      | $kv, k_{pe} = \text{split}(kv)$                           | $[N,L,512]\ [N,L,64] = \text{split}(N,L,576)$                     |
| $4$  | $\color{#1E90FF}{kv}=\text{norm}(kv)$                     | $[N,L,512] = \text{norm}([N,L,512])$                              |
| $5$  | $\color{#1E90FF}{k_{pe}} = \text{rotary\_emb}(k_{pe})$    | $[N,L,64] = \text{rotary\_emb}([N,L,64])$                         |
|      | $kv\_\text{cache} \leftarrow kv$                          | $[N,:\text{end\_pos},512]\triangleq [N, L',512]$                  |
|      | $pe\_\text{cache} \leftarrow k_{pe}$                      | $[N,:\text{end\_pos},64]\triangleq [N, L',64]$                    |
| $6$  | $q_{\text{nope}} = q_{\text{nope}} @ wkvb\_{\text{nope}}$ | $[N,L,128,512] = [N,L,128,128] @ [128, 128, 512]$                 |
| $7$  | $s_1=q_{\text{nope}}@kv\_\text{cache}$                    | $[N,L,128,L']= [N,L,128,512]@[N, L',512]$                         |
| $8$  | $s_2=q_{pe} @ pe\_\text{cache}$                           | $[N,L,128,L'] = [N,L,128,64]@[N,L',64]$                           |
| $9$  | $s = s_1 + s_2 $                                          | $[N,L,128,L']=[N,L,128,L']+[N,L,128,L']$                          |
| $10$ | $s = \text{softmax\_scale} * s$                           | $[N,L,128,L']=\text{softmax\_scale} * [N,L,128,L']$               |
| $11$ | $s = s+ \text{mask}$                                      | $[N,L,128,L'] = [N,L,128,L']@[L,1,L']$                            |
| $12$ | $s = \text{softmax}(s)$                                   | $[N,L,128,L']=\text{softmax}([N,L,128,L'])$                       |
| $13$ | $x = s @ kv\_\text{cache}$                                | $[N,L, 128, L'] @ [N,L',512] = [N,L,128,512]$                     |
| $14$ | $x = x @ wkvb\_{v}$                                       | $[N,L,128,128]=[N,L,128,512]@[128,128,512]$                       |
|      | $x = \text{flatten}(x, 2)$                                 | $[N,L,16384]=\text{flatten}([N,L,128,128])$                        |
| $15$ | $x = W_o @ x$                                             | $[N,L,7168]=[7168,16384]@[N,L,16384]$                             |

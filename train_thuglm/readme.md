# 训练`thuglm-6b`模型

# `thuglm-6b`模型和`gpt2`模型的差异

## loss部分

1. 查看了`thuglm-6b`模型源码，他的loss和`gpt2`等自回归模型的loss，基本上是一样的。(这里只是考虑自回归类型的训练)

```python
# 
# 这是thuglm模型的loss
if labels is not None:
    lm_logits = lm_logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(hidden_states.dtype)
    loss = loss.to(hidden_states.dtype)
```

```python
# src/transformers/models/gpt2/modeling_gpt2.py 的class GPT2LMHeadModel(GPT2PreTrainedModel):
# 这是gpt2的loss 
loss = None
if labels is not None:
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


```

## 代码风格

1. `thuglm-6b`源码和`transformers`包的`gpt2`源码，长得非常像，设计模式是一摸一样的。从工程角度来看，你只要看过`gpt2`
   的源码，看懂了，那么`thuglm-6b`的代码框架对你来说肯定不难。
2. 数学角度来说，这个我没有看过两个模型的论文，不敢胡说，这部分我就不解释了。

## 数据角度

1. `thuglm-6b`模型和`transformers`包的`gpt2`源码里面的模型，在`forward`方法里面，需要的参数，基本上是保持一致的，因此。需要的数据样式，也都差不多。
2. 那么虽然现在`thuglm-6b`还没有所谓的`thuglmForSequenceClassification`、`thuglmForTokenClassification`
   等方法，但是直接模仿`gpt2`的风格来写，就行了。就是`loss`更改一下，下游层更改一下。

## 本人对`thuglm-6b`模型的态度

1. `thuglm-6b`
   模型，最近太火了，而且在中文语言的表现上，效果非常好[https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
   ，使用int8还可以在小显存上进行推理，非常amazing。
2. 目前，很难在在市面上找到非常好的中文`gpt2`模型，可能是数据方面的问题，或者机器方面的问题。
3. 在我眼里，我其实就是把他当成一个在中文领域表现非常好的`gpt2`模型而已。（抛开别的条件不谈）。

# 训练`thuglm-6b`模型

| 序号  | 介绍                     | 文件夹                    | 是否已完成 | 是否还有bug |
|-----|------------------------|------------------------|-------|---------|
| 1   | 使用lora算法对`thuglm-6b`微调 | `v1_train_thuglm-lora` | ☑️    | ✅       |
| 2   | 单卡直接对`thuglm-6b`微调     | `v2_train_thuglm`      | ☑️    | ✅       |
| 3   | 多卡对`thuglm-6b`微调       | `v3_train_thuglm`      | ☑️    | ✅       |

# 1. 使用lora微调`thuglm-6b`模型

1.目前，训练一个`thuglm-6b`模型，还是比较费劲的（我还没试过，目前都在传使用lora方法来进行训练）。那也就跟风写一个教程。

2. 文本，将介绍如何使用`peft`[https://github.com/huggingface/peft](https://github.com/huggingface/peft)
   包（这个包实现了`lora`算法）、对`thuglm-6b`进行微调。
3. 硬件设备是3090（显存为24G）。
   4包括数据整理、模型转换、训练加载等详细步骤。


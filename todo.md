# TODO: Negotiator code

## Next steps
1. create generative network and predict words until eo conv
2. DONE. update training tf records
3. DONE. add input to model as context input
4. retrain

5. add output classifier
   - with final state input
   - with full dialogue and attention
6. add final output to loss
7. add outputs to generative network
8. retrain 

## Model
- DONE. encode agent input
- DONE. feed input into every dialogue step
- output classifier bidirectional with attention and input goals
- loss is word prediction loss + alpha * output loss

## Generation
- generate words from input until eos
- encode response until eos
- repeat
- finally generate outputs

## Goal based training
- reinforcement learning by conversing with fixed other agent
- reward: 0 if disagree, R = sum_x discounted (r - mean_r)
- backdate R to each word (*gamma)
- updates are R * grad_word
- grad_R = R * word_grad (REINFORCE)

## Rollout 
- rollout conversations to get max reward choice
- generate C candidtate sentences (until eos)
- roll out each C multiple times and average resulting reward
- weight rolled out R by probability of outputs
- example C=5, with 10 rollouts each

## Parameters
- input embedding 64
- word embedding 256
- input num_units (GRU) 64
- word num_units 128
- bidirectional output num units 256
- batch size 16 (SGD)
- initial lr 1.0
- Nesterov momentum 0.1
- clipping L^2 norm 0.5
- 30 epochs
- anneal lr by 5% each epoch
- loss alpha mixer 0.5
- rl learning rate 0.1
- rl clip gradients 1.0
- rl gamma 0.95
- 4 rl updates, 1 supervised (lr=0.5)
- sampling uses logits / 0.5 i.e. double sigmoid outputs

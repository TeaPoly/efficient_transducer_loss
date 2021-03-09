# efficient_transducer_loss
Transducer loss utterance by utterance.

> To eliminate such memory waste, we did not use the broadcasting method to combine h_enc and h_pre. Instead, we implement the combination sequence by sequence. Hence, the size of z_n for utterance n is T_n ∗ U_n ∗ D instead of max(T1,T2,,TN) ∗ max(U1, U2, , UN ) ∗ D. Then we concatenate all z_n instead of paralleling them, which means we convert z into a two-dimension tensor (sum(Tn ∗ Un)), D). In this way, the total memory cost for z1, z2, ...zN is (sum(Tn ∗ Un)) ∗ D. This significantly reduces the memory cost, compared to the broadcasting method.

# Reference

- Section 3.1. Efficient encoderand prediction output combination from IMPROVING RNN TRANSDUCER MODELING FOR END-TO-END SPEECH RECOGNITION (https://arxiv.org/pdf/1909.12415.pdf) 

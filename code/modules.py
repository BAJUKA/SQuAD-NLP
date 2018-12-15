# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class LSTMEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    This code uses a bidirectional LSTM, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("LSTMEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist



#co-attention as implemented in DCN paper

class CoAttn(object):

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """

        self.keep_probs = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size



    def build_graph(self, values, values_mask,keys_mask, keys):
        """
        Inputs:
           keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
           values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)
          keys_mask: Tensor shape (batch_size,num_keys)
            1s where there's a real input, 0's where there's padding

        Outputs:
            out: coattention context vector concated with document encoding.
        """
        with vs.variable_scope("CoAttn"):

            keys_t = tf.transpose(keys, perm=[0, 2, 1]) #(batch_size,value_vec_size,num_keys)
            values_t = tf.transpose(values,perm=[0,2,1]) #(batch_size,value_vec_size,num_values)

            #computing the affinity matrix 
            affinity_matrix = tf.matmul(keys,values_t) #(batch_size,num_keys,num_values)
            affinity_matrix_t = tf.transpose(affinity_matrix,perm=[0,2,1]) #(batch_size,num_values,num_keys)

            #normalizing the the affinity matrix and its transpose(i.e taking attention weights w.r.t questions and context)
            affinity_logits_mask_k = tf.expand_dims(keys_mask, 1) # shape (batch_size, 1, num_keys)
            affinity_logits_mask_v = tf.expand_dims(values_mask,1)# shape (batch_size,1,num_values)

            #The following are the operations given in the DCN paper to compute the coattention_context tensor
            _, A_Q = masked_softmax(affinity_matrix, affinity_logits_mask_v, 2) # shape (batch_size, num_keys,num_values). 
            _, A_D = masked_softmax(affinity_matrix_t, affinity_logits_mask_k, 2) # shape (batch_size, num_values,num_keys).

            C_Q = tf.matmul(keys_t,A_Q) #shape (batch_size,value_vec_size,num_keys)

            concated = tf.concat([values_t,C_Q],1)#shape (batch_size,2*value_vec_size,num_keys)
            coattention_context = tf.matmul(concated,A_D)#shape (batch_size,2*value_vec_size,num_keys)

            #concatinating the keys(document embedding with coattention context)
            out_t = tf.concat([keys_t,coattention_context],1) #shape (batch_size,3*value_vec_size,num_keys)

            out = tf.transpose(out_t,perm=[0,2,1]) #shape (batch_size,num_keys,3*value_vec_size)
            out = tf.nn.dropout(out,self.keep_probs)

            return(out)




class SelfAttn(object):
    '''
    Contains implemntation of question passage matching and passage passage matching as implemented in the R-Net paper
    '''
    def Create_Weight(self,name,size_inp,size_out):
        '''Creates a weight matrix of shape (size_inp,size_out) '''
        return tf.get_variable(name=name,shape=(size_inp,size_out),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

    def Create_Vector(self,name,size):
        #Creates a vector of shape (size)
        return tf.get_variable(name=name,shape=(size),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

    def Matrix_Multiplication(self,matrix,weight):
        matrix_shape = matrix.get_shape().as_list() #shape (batch_size,length,hidden_size)
        weight_shape = weight.get_shape().as_list() # shape(hidden_size,output_size)
        assert (matrix_shape[-1] == weight_shape[0])
        matrix_reshape = tf.reshape(matrix, [-1, matrix_shape[-1]]) #shape (batch_size*length,hidden_size)
        mul = tf.matmul(matrix_reshape, weight) #shape (batch_size*length,output_size)
        return tf.reshape(mul, [-1, matrix_shape[1], weight_shape[-1]]) #shape (batch_sizr,length,output_size)
        

    def __init__(self,hidden_size_encoder,hidden_size_qp,hidden_size_pp,keep_prob):

        self.keep_probs = keep_prob
        self.hidden_size_qp = hidden_size_qp
        self.hidden_size_pp = hidden_size_pp
        self.hidden_size_encoder = hidden_size_encoder

        #Weights for question passage matching

        self.W_v_p = self.Create_Weight("W_v_p",self.hidden_size_qp,self.hidden_size_qp)
        self.W_u_q = self.Create_Weight("W_u_q",2*self.hidden_size_encoder,self.hidden_size_qp)
        self.W_u_p = self.Create_Weight("W_u_p",2*self.hidden_size_encoder,self.hidden_size_qp)
        self.W_V = self.Create_Vector("W_V",self.hidden_size_qp)
        self.W_g = self.Create_Weight("W_g",self.hidden_size_encoder*4,self.hidden_size_encoder*4)

        #Weights for passage passage matching

        self.W_v_p_pp = self.Create_Weight("W_v_p_pp",self.hidden_size_qp,self.hidden_size_pp)
        self.W_v_qp_pp = self.Create_Weight("W_v_qp_pp",self.hidden_size_qp,self.hidden_size_pp)
        self.W_V_pp = self.Create_Vector("W_V_pp",self.hidden_size_pp)
        self.W_g_pp = self.Create_Weight("W_g_pp",self.hidden_size_qp*2,self.hidden_size_qp*2)

        #GRU cells for Question passage matching

        self.QP_GRU_Cell = tf.contrib.rnn.GRUCell(self.hidden_size_qp)
        self.QP_GRU_Cell = tf.contrib.rnn.DropoutWrapper(self.QP_GRU_Cell,self.keep_probs)
        

        #GRU cells for passage passage matching
        self.PP_GRU_Cell_f = tf.contrib.rnn.GRUCell(self.hidden_size_pp)
        self.PP_GRU_Cell_f = tf.contrib.rnn.DropoutWrapper(self.PP_GRU_Cell_f,self.keep_probs)

        self.PP_GRU_Cell_b = tf.contrib.rnn.GRUCell(self.hidden_size_pp)
        self.PP_GRU_Cell_b = tf.contrib.rnn.DropoutWrapper(self.PP_GRU_Cell_b,self.keep_probs)

    def build_graph_qp(self,context_encoding,question_encoding,context_mask,question_mask,context_len,question_len):
        #implementation of question passage matching

        u_q = question_encoding #(batch_size,question_len,2*hidden_size_encoder)
        u_p = context_encoding #(batch_size,context_len,2*hidden_size_encoder)
        vP = []

        for i in range(context_len):
            question_rep = self.Matrix_Multiplication(u_q,self.W_u_q) #(batch_size,question_len,hidden_size_qp)
            #print(question_rep.shape)
            curr_batch_size = tf.shape(context_encoding)[0]


            u_p_i = tf.reshape(u_p[:,i,:],[curr_batch_size,1,2*self.hidden_size_encoder])
            passage_wrd_rep = self.Matrix_Multiplication(u_p_i,self.W_u_p) #(batch_size,1,hidden_size_qp)

            if i==0:
                tanh_qp = tf.tanh(question_rep + passage_wrd_rep)
            else:
                slice_vP = tf.reshape(vP[i-1],[curr_batch_size,1,self.hidden_size_qp])
                vP_i = self.Matrix_Multiplication(slice_vP,self.W_v_p)
                tanh_qp = tf.tanh(vP_i + question_rep + passage_wrd_rep) #(batch_size,question_len,hidden_size_qp)

            V_qp = self.Matrix_Multiplication(tanh_qp,tf.reshape(self.W_V,[-1,1])) #(batch_size,question_len,1)
            V_qp = tf.squeeze(V_qp,axis=2) #(batch_size,question_len)

            _,softmax_v_qp = masked_softmax(V_qp,question_mask,1) #(batch_size,question_len)
            softmax_v_qp = tf.expand_dims(softmax_v_qp,axis=1) #(batch_size,1,question_len)

            question_aware_rep = tf.reduce_sum(tf.matmul(softmax_v_qp,u_q),1) #(batch_size,2*hidden_size_encoder)

            slice_q_p = u_p[:,i,:] #(batch_size,2*hidden_size_encoder)
            concated_qp = tf.concat([slice_q_p,question_aware_rep],1) #(batch_size,4*hidden_size_encoder)

            gate_rep_qp = tf.sigmoid(tf.matmul(concated_qp,self.W_g))  #(batch_size,4*hidden_size_encoder)
            final_gate_rep_qp = tf.multiply(gate_rep_qp,concated_qp) #(batch_size,4*hidden_size_encoder)



            self.QP_GRU_Cell_state = self.QP_GRU_Cell.zero_state(batch_size=curr_batch_size, dtype=tf.float32)

            with tf.variable_scope("QP_attention"):
                if i > 0: tf.get_variable_scope().reuse_variables()
                output, self.QP_GRU_Cell_state = self.QP_GRU_Cell(final_gate_rep_qp, self.QP_GRU_Cell_state)
                vP.append(output)

        vP = tf.stack(vP, 1)
        vP = tf.nn.dropout(vP,self.keep_probs)

        return vP




        

    def build_graph_pp(self,context_encoding,question_encoding,context_mask,question_mask,v_p,context_len,question_len):
        #implementation of passage passage matching

        u_q = question_encoding #[batch_size, question_len, hidden_size*2]
        u_p = context_encoding #[batch_size, context_len, hidden_size*2]

        v_pp = []
        #print(v_p.shape)

        for i in range(context_len): 
            passage_rep = self.Matrix_Multiplication(v_p,self.W_v_p_pp) #(batch_size,context_len,hidden_size_pp)
            #print(passage_rep.shape)
            curr_batch_size = tf.shape(v_p)[0]

            v_p_i = tf.reshape(v_p[:,i,:],[curr_batch_size,1,self.hidden_size_qp])
            passage_word_rep = self.Matrix_Multiplication(v_p_i,self.W_v_qp_pp) #(batch_size,1,hiddden_size_pp)
            #print(passage_word_rep.shape)
            tanh_pp = tf.tanh(passage_rep + passage_word_rep) #(batch_size,context_len,hidden_size_pp)
            #print(tanh_pp.shape)
            V_pp = self.Matrix_Multiplication(tanh_pp,tf.reshape(self.W_V_pp,[-1,1])) #(batch_size,context_len,1)
            V_pp = tf.squeeze(V_pp,axis=2) #(batch_size,context_len)

            _,softmax_V_pp = masked_softmax(V_pp,context_mask,1) #(batch_size,context_len)
            softmax_V_pp = tf.expand_dims(softmax_V_pp,axis=1) #(batch_size,1,context_len)
            #print(softmax_V_pp.shape)


            passage_aware_rep = tf.reduce_sum(tf.matmul(softmax_V_pp,v_p),1) #(batch_size,hidden_size_qp)
            #print(passage_aware_rep.shape)

            slice_v_p = v_p[:,i,:] #(batch_size,hiddden_size_pp)
            #print(slice_v_p.shape)

            concated = tf.concat([slice_v_p,passage_aware_rep],1) #(batch_size,2*hidden_size)

            gate_rep = tf.sigmoid(tf.matmul(concated,self.W_g_pp)) #(batch_size,2*hidden_size)

            final_gate_rep = tf.multiply(gate_rep,concated) #(batch_size,2*hidden_size)

            v_pp.append(final_gate_rep)

        v_pp = tf.stack(v_pp, 1)
        unstacked_v_pp = tf.unstack(v_pp,context_len,1)

        self.pp_fw_state = self.PP_GRU_Cell_f.zero_state(curr_batch_size,dtype=tf.float32)
        self.pp_bw_state = self.PP_GRU_Cell_b.zero_state(curr_batch_size,dtype=tf.float32)

        with tf.variable_scope('Self_match') as scope:
            pp_outputs, pp_forward, pp_backward = tf.contrib.rnn.static_bidirectional_rnn(self.PP_GRU_Cell_f, self.PP_GRU_Cell_b,unstacked_v_pp,dtype=tf.float32)
            h_p = tf.stack(pp_outputs, 1)

        h_p = tf.nn.dropout(h_p,self.keep_probs)

        return(h_p)


class AnswerPointer(object):
    '''
    Contains the implementation of answer pointer as implemented in the R-Net paper
    '''
    def Create_Weight(self,name,size_inp,size_out):
        '''Creates a weight matrix of shape (size_inp,size_out) '''
        return tf.get_variable(name=name,shape=(size_inp,size_out),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

    def Create_Vector(self,name,size):
        #Creates a vector of shape (size)
        return tf.get_variable(name=name,shape=(size),dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

    def Matrix_Multiplication(self,matrix,weight):
        matrix_shape = matrix.get_shape().as_list() #shape (batch_size,length,hidden_size)
        weight_shape = weight.get_shape().as_list() # shape(hidden_size,output_size)
        assert (matrix_shape[-1] == weight_shape[0])
        matrix_reshape = tf.reshape(matrix, [-1, matrix_shape[-1]]) #shape (batch_size*length,hidden_size)
        mul = tf.matmul(matrix_reshape, weight) #shape (batch_size*length,output_size)
        return tf.reshape(mul, [-1, matrix_shape[1], weight_shape[-1]]) #shape (batch_sizr,length,output_size)

    def __init__(self,hidden_size_encoder,hidden_size_attn,question_len,keep_prob):
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_attn = hidden_size_attn
        self.question_len = question_len
        self.keep_probs = keep_prob

        #Weights for question pooling

        self.W_u_q_ap = self.Create_Weight('W_u_q_ap',2*self.hidden_size_encoder,self.hidden_size_encoder)
        self.W_v_q_ap = self.Create_Weight('W_v_q_ap',self.hidden_size_encoder,self.hidden_size_encoder)
        self.W_v_r_ap = self.Create_Weight('W_v_r_ap',self.question_len,self.hidden_size_encoder)
        self.V_Q_P_ap = self.Create_Vector('V_Q_P_ap',self.hidden_size_encoder)

        #Weights for answer pointer

        self.W_h_ap = self.Create_Weight('W_h_ap',2*self.hidden_size_encoder,2*self.hidden_size_encoder)
        self.W_pr_ap = self.Create_Weight('W_pr_ap',self.hidden_size_attn,2*self.hidden_size_encoder)
        self.W_v_ap = self.Create_Vector('W_v_ap',2*self.hidden_size_encoder)

        self.ap_GRU = tf.contrib.rnn.GRUCell(2*self.hidden_size_encoder)
        self.ap_GRU = tf.contrib.rnn.DropoutWrapper(self.ap_GRU,input_keep_prob=self.keep_probs)

    def QuesPool(self,question_encoding,question_mask):
        u_q = question_encoding #(batch_size,question_len,2*hidden_size_encoder)

        question_repr = self.Matrix_Multiplication(u_q,self.W_u_q_ap) #(batch_size,question_len,hidden_size_encoder)
        bias_repr = tf.matmul(self.W_v_r_ap,self.W_v_q_ap) #(question_len,hidden_size_encoder)
        cur_batch_size = tf.shape(u_q)[0]

        bias_repr = tf.expand_dims(bias_repr,axis=0) #(1,question_len,hidden_size_encoder)

        combined_repr = tf.tanh(question_repr+bias_repr) #(batch_size,question_len,hidden_size_encoder)

        pre_attention = self.Matrix_Multiplication(combined_repr,tf.reshape(self.V_Q_P_ap,[-1,1])) #(batch_size,question_len,1)

        pre_attention = tf.squeeze(pre_attention,axis=2)

        _,masked_softmax_rep = masked_softmax(pre_attention,question_mask,1) #(batch_size,question_length)
        masked_softmax_rep = tf.expand_dims(masked_softmax_rep,axis=1) #(batch_size,1,question_len)

        r_q = tf.reduce_sum(tf.matmul(masked_softmax_rep,u_q),1) #(batch_size,2*hidden_size_encoder)
        r_q = tf.nn.dropout(r_q, self.keep_probs)

        return r_q

    def build_graph_answer_pointer(self,question_encoding,context_encoding,h_p,question_length,context_len,question_mask,context_mask):

        ques_pool = self.QuesPool(question_encoding,question_mask)
        ha = None
        cur_batch_size = tf.shape(question_encoding)[0]
        p = []
        logits = []
        #print(ques_pool.shape)
        for i in range(2):
            encoded_pass_rep = self.Matrix_Multiplication(h_p,self.W_pr_ap) #(batch_size,context_len,2*hidden_size,encoder)
            #print(i)
            if i==0:
                h_i = ques_pool
            else:
                h_i = ha
            #print(h_i.shape)
            concat_h_i = tf.concat([tf.reshape(h_i, [cur_batch_size, 1, 2*self.hidden_size_encoder])] * context_len, 1)
            prev_encodind_rep = self.Matrix_Multiplication(concat_h_i,self.W_h_ap) #(batch_size,context_len,2*hidden_size_encoder)

            tanh = tf.tanh(encoded_pass_rep+prev_encodind_rep) #(batch_size,context_len,2*hidden_size,encoder)

            pre_attention = self.Matrix_Multiplication(tanh,tf.reshape(self.W_v_ap,[-1,1])) #(batch_size,context_len,1)
            pre_attention = tf.squeeze(pre_attention,axis=2) #(batch_size,context_len)

            logits_ptr, a_i_ptr = masked_softmax(pre_attention,context_mask,1) #(batch_size,context_len)

            logits.append(logits_ptr)
            p.append(a_i_ptr)

            a_i_ptr = tf.expand_dims(a_i_ptr,axis=1)#(batch_size,1,context_len)
            attention_rep = tf.reduce_sum(tf.matmul(a_i_ptr,h_p),1) #(batch_size,hidden_size_attention)

            
            if i==0:
                self.ap_GRU_state = self.ap_GRU.zero_state(batch_size=cur_batch_size,dtype=tf.float32)
                ha,_ = self.ap_GRU(attention_rep,self.ap_GRU_state)

        return p,logits



























            

        








# Question Answering on SQuAD
I started this project to explore the domain of Question Answering especially attention mechanism and how they affect the overall performance of the model. SQuAD is a reading comprehension data set. This means a paragraph and question about the paragraph as input to the model. The answer to the question is a continuous span in the paragraph i.e we have to predict the start and end indices of the answer. More info about the data set can be found in <b>data_visualization.py</b> and also in the paper <a>https://arxiv.org/pdf/1606.05250.pdf</a>. I have used the template provided for the default final project of Stanford's CS224n course.

## Papers(parts of it) I have tried to implement
* [DYNAMIC COATTENTION NETWORKS FOR QUESTION ANSWERING](https://arxiv.org/abs/1611.01604)(only the coattention part)
* [R-NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
* [Smart-Span at test time](https://arxiv.org/pdf/1704.00051.pdf)<br>

This blog helped me to gain a lot of insight about the R-NET [Challenges of reproducing R-NET](https://yerevann.github.io/2017/08/25/challenges-of-reproducing-r-net-neural-network-using-keras/)<br>

## About the code

 *  **get_started.sh** : A script to install requirements, and download and preprocess the data.<br>
 * **requirements.txt**: Used by get_started.sh to install requirements
* **code/**: A directory containing all code:<br>
**– preprocessing/**: Code to preprocess the SQuAD data, so it is ready for training:<br>
* **download_wordvecs.py**: Downloads and stores the pretrained word vectors (GloVe).<br>
* **squad_preprocess.py**: Downloads and preprocesses the official SQuAD train and dev sets and writes the preprocessed versions to file.<br>
* **data_batcher.py**: Reads the pre-processed data from file and processes it into batches for training.<br>
* **main.py**: The top-level entrypoint to the code. You can run this file to train the model, view examples from the model and evaluate the model.<br>
* **modules.py**: Contains componets for different models.<br>
* **pretty_print.py**: Contains code to visualize model output.<br>
* **qa_model.py**: Contains the model definition. <br>
* **vocab.py**: Contains code to read GloVe embeddings from file and make into an embedding matrix.<br>

## Results
I have tried to experiment with things I have learnt and tried to understand their effect on the performance as much as I can.
* The various models were compared against a baseline which had a RNN Encoder with simple attention and a softmax decoder which gave a F1 score of about 40% on the dev set.<br>
* With coattention and softmax decoder I managed to get a F1 score of 63% on the dev set.<br>
* With self attention(R-NET) and softmax decoder I managed to get a F1 score of 62% on the dev set.<br>
* With the complete R-NET architecture (i.e self attention and answer pointer) and with co-attention and answer pointer I managed to get an F1 score of 64%. <br>
These are some of the images of the question answered by the model along with the right answer and F1 score.
![Result1](https://github.com/BAJUKA/SQuAD-NLP/blob/master/screenshot/result.png)
![Result2](https://github.com/BAJUKA/SQuAD-NLP/blob/master/screenshot/result2.png)
![Result3](https://github.com/BAJUKA/SQuAD-NLP/blob/master/screenshot/result3.png)

## Future Work
* Analyze the results produced by the model to understand where it has gone wrong so that I can make changes to improve on these(like smart span!).
* Improve the R-NET model as I am not satisfied with the F1 score. The fault may be in the implementation of the answer pointer which is not giving the expected results.
* Experiment more. I have just started exploring the question answering domain and want to implement and experiment with more papers as I read them.

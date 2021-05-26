import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    '''
    Encoder helps in building the sentence encoding module for a batched version
    of data that is sent in [T x B] having corresponding input lengths in [1 x B]

    Args:
            hidden_size: Hidden size of the RNN cell
            embedding: Embeddings matrix [vocab_size, embedding_dim]
            cell_type: Type of RNN cell to be used : LSTM, GRU
            nlayers: Number of layers of LSTM (default = 1)
            dropout: Dropout Rate (default = 0.1)
            bidirectional: Bidirectional model to be formed (default: False)
    '''

    def __init__(self, hidden_size=512,embedding_size = 768, cell_type='lstm', nlayers=1, dropout=0.1, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.cell_type = cell_type
        self.embedding_size = embedding_size
        # self.embedding_size = self.embedding.embedding_dim
        self.bidirectional = bidirectional

        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
                               num_layers=self.nlayers,
                               dropout=(0 if self.nlayers == 1 else dropout),
                               bidirectional=bidirectional)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size,
                              num_layers=self.nlayers,
                              dropout=(0 if self.nlayers == 1 else dropout),
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(self.embedding_size, self.hidden_size,
                              num_layers=self.nlayers,
                              nonlinearity='tanh',                            # ['relu', 'tanh']
                              dropout=(0 if self.nlayers == 1 else dropout),
                              bidirectional=bidirectional)

    def forward(self, sorted_seqs, sorted_len, orig_idx, device=None, hidden=None):
        '''
            Args:
                input_seqs (tensor) : input tensor | size : [Seq_len X Batch_size]
                input_lengths (list/tensor) : length of each input sentence | size : [Batch_size] 
                device (gpu) : Used for sorting the sentences and putting it to device

            Returns:
                output (tensor) : Last State representations of RNN [Seq_len X Batch_size X hidden_size]
                hidden (tuple)    : Hidden states and (cell states) of recurrent networks
        '''

        # sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seqs, input_lengths, device)
        # pdb.set_trace()

        #embedded = self.embedding(sorted_seqs)  ### NO MORE IDS
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_seqs, sorted_len)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            outputs)  # unpack (back to padded)

        outputs = outputs.index_select(1, orig_idx)

        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

        return outputs, hidden


class DecoderRNN(nn.Module):
    '''
    To DO
    Encoder helps in building the sentence encoding module for a batched version
    of data that is sent in [T x B] having corresponding input lengths in [1 x B]

    Args:
            hidden_size: Hidden size of the RNN cell
            embedding: Embeddings matrix [vocab_size, embedding_dim]
            cell_type: Type of RNN cell to be used : LSTM, GRU
            nlayers: Number of layers of LSTM (default = 1)
            dropout: Dropout Rate (default = 0.1)
            bidirectional: Bidirectional model to be formed (default: False)
    '''
    def __init__(self, embedding, cell_type, hidden_size, output_size, nlayers=1, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.hidden_size        = hidden_size
        self.cell_type          = cell_type
        self.embedding          = embedding
        self.embedding_size     = self.embedding.embedding_dim
        self.embedding_dropout = nn.Dropout(dropout)
        self.nlayers            = nlayers
        self.output_size        = output_size

        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.nlayers, dropout=(0 if nlayers == 1 else dropout))
        else:
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, num_layers=self.nlayers, dropout=(0 if nlayers == 1 else dropout))

        self.out     = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input_step, last_hidden):
        '''
        To Do
            Args:
                input_seqs (tensor) : input tensor | size : [Seq_len X Batch_size]
                input_lengths (list/tensor) : length of each input sentence | size : [Batch_size] 
                device (gpu) : Used for sorting the sentences and putting it to device

            Returns:
                output (tensor) : Last State representations of RNN [Seq_len X Batch_size X hidden_size]
                hidden (tuple)    : Hidden states and (cell states) of recurrent networks
        '''
        output              = self.embedding(input_step)
        output              = self.embedding_dropout(output)
        output              = output.view(1, input_step.size(0), self.embedding_size)
        output              = F.relu(output)
        output, last_hidden = self.rnn(output, last_hidden)
        output              = output.squeeze(0)
        output              = self.out(output)
        output              = F.log_softmax(output, dim=1)

        return output, last_hidden


# Luong attention layer
class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, "is not an appropriate attention method.")
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

	def dot_score(self, hidden, encoder_outputs):
		return torch.sum(hidden * encoder_outputs, dim=2)

	def general_score(self, hidden, encoder_outputs):
		energy = self.attn(encoder_outputs)
		return torch.sum(hidden * energy, dim=2)

	def concat_score(self, hidden, encoder_outputs):
		energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), 2)).tanh()
		return torch.sum(self.v * energy, dim=2)

	def forward(self, hidden, encoder_outputs):
		# Calculate the attention weights (energies) based on the given method
		if self.method == 'general':
			attn_energies = self.general_score(hidden, encoder_outputs)
		elif self.method == 'concat':
			attn_energies = self.concat_score(hidden, encoder_outputs)
		elif self.method == 'dot':
			attn_energies = self.dot_score(hidden, encoder_outputs)

		# Transpose max_length and batch_size dimensions
		attn_energies = attn_energies.t()

		# Return the softmax normalized probability scores (with added dimension)
		return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, embedding, cell_type, hidden_size, output_size, nlayers=1, dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model 	= attn_model
		self.hidden_size 	= hidden_size
		self.output_size 	= output_size
		self.nlayers 		= nlayers
		self.dropout 		= dropout
		self.cell_type 		= cell_type

		# Define layers
		self.embedding = embedding
		self.embedding_size  = self.embedding.embedding_dim
		self.embedding_dropout = nn.Dropout(self.dropout)
		if self.cell_type == 'gru':
			self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.nlayers, dropout=(0 if self.nlayers == 1 else self.dropout))
		else:
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.nlayers, dropout=(0 if self.nlayers == 1 else self.dropout))
		self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

		self.attn = Attn(self.attn_model, self.hidden_size)

	def forward(self, input_step, last_hidden, encoder_outputs):
		# Note: we run this one step (word) at a time
		# Get embedding of current input word
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)

		try:
			embedded = embedded.view(1, input_step.size(0), self.embedding_size)
		except:
			embedded = embedded.view(1, 1, self.embedding_size)

		rnn_output, hidden = self.rnn(embedded, last_hidden)
		# Calculate attention weights from the current GRU output
		attn_weights = self.attn(rnn_output, encoder_outputs)
		# Multiply attention weights to encoder outputs to get new "weighted sum" context vector
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		# Concatenate weighted context vector and GRU output using Luong eq. 5
		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = F.relu(self.concat(concat_input))
		representation = concat_output
		# Predict next word using Luong eq. 6
		output = self.out(concat_output)
		output = F.log_softmax(output, dim=1)
		# Return output and final hidden state
		return output, hidden, attn_weights, representation
		
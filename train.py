from load_data import *
from mt import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

def trainer(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    #,  max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #import pdb; pdb.set_trace()
    batch_size = input_variable.size(1)

    input_length = input_variable.size(0)
    target_length = target_variable.size(0)
    input_variable = Variable(input_variable)
    target_variable = Variable(target_variable)

    encoder_outputs = Variable(torch.zeros(input_length, batch_size, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
   
    loss = 0

    
    encoder_outputs, encoder_hidden = encoder(
            input_variable, encoder_hidden)
    """
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]
    """

    #import pdb; pdb.set_trace()

    #decoder_input = Variable(torch.LongTensor([[0]])) # 0 = <SOS>
    decoder_input = Variable(torch.zeros(batch_size).long())
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden
    max_length = encoder_outputs.size(0)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    decoded_outs = []
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, max_length)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
            topv, topi = decoder_output.data.topk(1)
            decoded_outs.append(topi.squeeze())

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, max_length)
            topv, topi = decoder_output.data.topk(1)
            decoded_outs.append(topi.squeeze())
            #ni = topi[0][0]

            #decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = Variable(topi.squeeze())
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            
            loss += criterion(decoder_output, target_variable[di])
            #if ni == 1: # <EOS>
            #    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length, decoded_outs

def accuracy(output, target):
    bz = target.size(1)
    length = target.size(0)
    output = torch.stack(output, 0).chunk(length, dim=1)
    target = target.chunk(length, dim=1)

    acc = 0
    for i in range(bz):
        if torch.eq(target[0], output[0]).all():
            acc += 1

    return acc / bz

def trainIters(encoder, decoder, train_iter, n_iters, print_every=100, plot_every=100, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    acc = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #training_pairs = [variablesFromPair(random.choice(pairs))
    #                  for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = get_batch(next(train_iter))
        #training_pair = training_pairs[iter - 1]
        input_variable = torch.cat(training_pair[0], dim=1)
        target_variable = torch.cat(training_pair[1], dim=1)
        loss, decoded_outs = trainer(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        acc += accuracy(decoded_outs, target_variable)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f %.2f' % (iter, iter / n_iters * 100, print_loss_avg, (acc / iter)))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)

def evaluate(encoder, decoder, sentence):#, max_length=MAX_LENGTH):
    """
    Function that generate translation.
    First, feed the source sentence into the encoder and obtain the hidden states from encoder.
    Secondly, feed the hidden states into the decoder and unfold the outputs from the decoder.
    Lastly, for each outputs from the decoder, collect the corresponding words in the target language's vocabulary.
    And collect the attention for each output words.
    @param encoder: the encoder network
    @param decoder: the decoder network
    @param sentence: string, a sentence in source language to be translated
    @param max_length: the max # of words that the decoder can return
    @output decoded_words: a list of words in target language
    @output decoder_attentions: a list of vector, each of which sums up to 1.0
    """
    # process input sentence

    ### FIIIXX

    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size(0)
    target_length =  target_variable.size(0)
    
    # encode the source lanugage
    encoder_hidden = encoder.initHidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    
    # decode the context vector
    decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    # output of this function
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    # unfold
    for di in range(max_length):
        # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        # hint: print out decoder_output and decoder_attention
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        # stop unfolding whenever '<EOS>' token is returned
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


train = load_data("add_prim_split/tasks_train_addprim_jump.txt")
inn, outt = build_dictionary(train)
train_iter = data_iter(train, 10)


teacher_forcing_ratio = 0.0


hidden_size = 50
encoder1 = EncoderRNN(len(inn[2]), hidden_size)
decoder1 = DecoderRNN(hidden_size, len(outt[2]), 1)#, dropout_p=0.1)
#decoder1 = AttnDecoderRNN(hidden_size, len(outt[2]), 1)#, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = decoder1.cuda()

trainIters(encoder1, decoder1, train_iter, 750000, print_every=100)

torch.save(encoder1.state_dict(), "saved_encoder.pth")
torch.save(attn_decoder1.state_dict(), "saved_decoder.pth")


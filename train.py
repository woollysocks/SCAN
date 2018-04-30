from load_data import *
from mt import *
import logger

import gflags
import sys
import os

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

    
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
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

def evaluater(input_variable, encoder, decoder, max_length=49):
    #print_loss_total = 0  # Reset every print_every
    #acc = 0
    #for i in range(len(eval_iters)):
    encoder.eval()
    decoder.eval()
    input_length = input_variable.size(0)
    input_variable = Variable(input_variable)

    #encoder_outputs = Variable(torch.zeros(input_length, batch_size, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    encoder_hidden = encoder.initHidden()
    
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    decoder_input = Variable(torch.zeros(batch_size).long())
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoded_outs = []
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, max_length)
        topv, topi = decoder_output.data.topk(1)
        decoded_outs.append(topi.squeeze())

        decoder_input = Variable(topi.squeeze())
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_outs

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

def save_checkpoint(state, filename):
    torch.save(state, filename)

def trainIters(encoder, decoder, train_iter, eval_iters, n_iters, print_every=100, eval_every=1000, learning_rate=0.001, exp_name=None, logger=None, path=None):

    print_loss_total = 0  # Reset every print_every
    acc = 0
    best_step = 0
    best_eval_acc = 0
    start_iter = 1

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    #training_pairs = [variablesFromPair(random.choice(pairs))
    #                  for i in range(n_iters)]
    criterion = nn.NLLLoss()

    ckpt_path = path
    ckpt_name = path + exp_name + ".ckpt"
    ckpt_best = path + ckpt_name + "_best"

    if os.path.isfile(ckpt_name):
        print("=> loading checkpoint '{}'".format(ckpt_name))
        checkpoint = torch.load(ckpt_name)
        start_iter = checkpoint['iter']
        best_eval_acc = checkpoint['best_eval']
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        print("=> loaded checkpoint '{}' (step {})"
              .format(ckpt_name, start_iter))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_name))

    for iter in range(start_iter, n_iters + 1):
        training_pair = get_batch(next(train_iter))
        #training_pair = training_pairs[iter - 1]
        input_variable = torch.cat(training_pair[0], dim=1)
        target_variable = torch.cat(training_pair[1], dim=1)
        loss, decoded_outs = trainer(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        acc += accuracy(decoded_outs, target_variable)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            logger.Log('Step: (%d %d%%) \t Loss: %.5f \t Acc: %.4f' % (iter, iter / n_iters * 100, print_loss_avg, (acc / iter)))

        if iter % eval_every == 0:
            eval_acc = 0
            for i in range(len(eval_iters)):
                eval_pair = get_batch(eval_iters[i])
                #import pdb; pdb.set_trace()
                input_variable = torch.cat(eval_pair[0], dim=1)
                target_variable = torch.cat(eval_pair[1], dim=1)
                decoded_outs = evaluater(input_variable, encoder, decoder)
                eval_acc += accuracy(decoded_outs, target_variable)
            
            save_checkpoint({
                    'iter': iter,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'best_eval': best_eval_acc,
                    'encoder_optimizer' : encoder_optimizer.state_dict(),
                    'decoder_optimizer' : decoder_optimizer.state_dict(),
                    }, filename=ckpt_name)

            eval_acc = eval_acc / len(eval_iters)
            logger.Log('Eval acc: %.4f' % (eval_acc))

            if eval_acc - 0.0020 > best_eval_acc:
                best_eval_acc = eval_acc
                encoder.save(encoder)
                save_checkpoint({
                    'iter': iter,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'best_eval': best_eval_acc,
                    'encoder_optimizer' : encoder_optimizer.state_dict(),
                    'decoder_optimizer' : decoder_optimizer.state_dict(),
                    }, filename=ckpt_best)


FLAGS = gflags.FLAGS
gflags.DEFINE_string("log_path", "./logs/", "Folder with log files.")
gflags.DEFINE_string("exp_name", "", "Experiment name")
gflags.DEFINE_integer("batch_size", "10", "Batch_size")
gflags.DEFINE_float("drop_out", "0.1", "Drop out rate")
gflags.DEFINE_integer("num_layers", "1", "Number of layers")
gflags.DEFINE_enum("decoder",
                       "standard",
                       ["standard",
                        "attention"],
                       "Type of decoder.")

gflags.DEFINE_string("eval_data", "add_prim_split/tasks_test_addprim_jump.txt", "Path to eval data")
gflags.DEFINE_string("train_data", "add_prim_split/tasks_train_addprim_jump.txt", "Path to training data")

gflags.DEFINE_string("train_data_supp0", "None", "Path to training data supplement 0")
gflags.DEFINE_string("train_data_supp1", "None", "Path to training data supplement 1")
gflags.DEFINE_string("train_data_supp2", "None", "Path to training data supplement 2")
gflags.DEFINE_string("train_data_supp3", "None", "Path to training data supplement 3")

FLAGS(sys.argv)

exp_name = FLAGS.exp_name
logpath = FLAGS.log_path + FLAGS.exp_name + ".log"
logger = logger.Logger(logpath)
batch_size = FLAGS.batch_size

train = load_data(FLAGS.train_data)
test = load_data(FLAGS.eval_data)
train_in, train_out = indexify_and_build_dictionary(train)
test_in, test_out = indexify_and_build_dictionary(test)
train_iter = data_iter(train, batch_size)
eval_iters = eval_iter(test, batch_size)


teacher_forcing_ratio = 0.0


hidden_size = 50
encoder1 = EncoderRNN(len(train_in[2]), hidden_size)

if FLAGS.decoder == "standard":
    decoder1 = DecoderRNN(hidden_size, len(train_out[2]), 1)#, dropout_p=0.1)
else:
    decoder1 = AttnDecoderRNN(hidden_size, len(outt[2]), 1, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = decoder1.cuda()

trainIters(encoder1, decoder1, train_iter, eval_iters, 750000, print_every=100, eval_every=1000, exp_name=exp_name, logger=logger, path=FLAGS.log_path)

torch.save(encoder1.state_dict(), "saved_encoder.pth")
torch.save(attn_decoder1.state_dict(), "saved_decoder.pth")


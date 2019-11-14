import argparse
import torch
import os
import utils
import modules

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100,
                    help='Number of training iterations.')
parser.add_argument('--learning-rate', type=float, default=1e-2,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=32,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=512,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-symbols', type=int, default=5,
                    help='Number of distinct symbols in data generation.')
parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--log-interval', type=int, default=5,
                    help='Logging interval.')
parser.add_argument('--log-dir', type=str, default="output",
                    dest="log_dir",
                    help='Logging directory.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

model = modules.CompILE(
    input_dim=args.num_symbols + 1,  # +1 for EOS/Padding symbol.
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Create directory for plots
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
# Create file for plot
plt_fname = os.path.join(args.log_dir,  '_'.join([str(x) for x in [
    args.num_symbols,"symbols",
    args.num_segments,"segments",
    args.latent_dim, "latent",
    args.latent_dist,
    args.iterations, "iters.txt"]])
)
with open(plt_fname, 'w') as plotf:
    plotf.write("step,nll,rec_acc,batch_nll,batch_rec_acc\n")

# Train model.
print('Training model...')
for step in range(args.iterations):
    data = None
    rec = None
    batch_loss = 0
    batch_acc = 0
    optimizer.zero_grad()

    # Generate data.
    data = []
    for _ in range(args.batch_size):
        data.append(utils.generate_toy_data(
            num_symbols=args.num_symbols,
            num_segments=args.num_segments))
    lengths = torch.tensor(list(map(len, data)))
    lengths = lengths.to(device)

    inputs = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    inputs = inputs.to(device)

    # Run forward pass.
    model.train()
    outputs = model.forward(inputs, lengths)
    loss, nll, kl_z, kl_b = utils.get_losses(inputs, outputs, args)

    loss.backward()
    optimizer.step()

    if step % args.log_interval == 0:
        # Run evaluation.
        model.eval()
        outputs = model.forward(inputs, lengths)
        acc, rec = utils.get_reconstruction_accuracy(inputs, outputs, args)

        # Accumulate metrics.
        batch_acc += acc.item()
        batch_loss += nll.item()
        print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(
            step, batch_loss, batch_acc))
        print('input sample: {}'.format(inputs[-1, :lengths[-1] - 1]))
        print('reconstruction: {}'.format(rec[-1]))
        # Append results to file
        with open(plt_fname, 'a') as pltf:
            pltf.write(','.join([str(x) for x in [
                step, nll.item(), acc.item(), 
                batch_loss, batch_acc
            ]]) + "\n")


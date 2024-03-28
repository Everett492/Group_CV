import jittor as jt
import os
from jittor import nn, Module
import jittor.transform as transforms
import numpy as np
from tqdm import tqdm
from jittor.dataset import CIFAR10
from jittor_attention import MultiheadAttention
import time

data_root = '~/.cache/'
batch_size = 64
lr = 1e-3
momentum = 0.9
epochs = 200

jt.flags.use_cuda = 1

transform = transforms.Compose([
    # transforms.RandomCrop(28),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(size=(256, 256)),
    transforms.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Use Cifar100 dataset
train_loader = CIFAR10(os.path.expanduser(data_root), train=True, transform=transform, download=True).set_attrs(batch_size=batch_size, shuffle=True)
test_loader = CIFAR10(os.path.expanduser(data_root), train=False, transform=transform, download=True).set_attrs(shuffle=True)

class PositionalEmbedding(Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(jt.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def execute(self, x):
        output = x + self.pos_embedding
        if self.dropout:
            output = self.dropout(output)
        return output

class MLPBlock(Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MLPBlock, self).__init__()

        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def execute(self, x):
        output = self.fc1(x)
        output = self.act(output)
        if self.dropout1:
            output = self.dropout1(output)
        output = self.fc2(output)
        if self.dropout2:
            output = self.dropout2(output)
        
        return output

class EncoderBlock(Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=attn_dropout_rate, batch_first=True)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MLPBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def execute(self, x):
        residual = x
        output = self.norm1(x)
        output = self.attn(output, output, output)
        if self.dropout:
            output = self.dropout(output[0])
        output += residual
        residual = output

        output = self.norm2(output)
        output = self.mlp(output)
        output += residual

        return output

class Encoder(Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()
        
        self.pos_embdding = PositionalEmbedding(num_patches, emb_dim, dropout_rate)

        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)
    
    def execute(self, x):
        output = self.pos_embdding(x)
        
        for layer in self.encoder_layers:
            output = layer(output)

        output = self.norm(output)

        return output

class VisionTransformer(Module):
    def __init__(self,
                 img_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=10,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = img_size

        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))

        # class token
        self.cls_token = nn.Parameter(jt.zeros(1, 1, emb_dim))

        # transfromer encoder
        self.encoder = Encoder(
            num_patches,
            emb_dim,
            mlp_dim,
            num_layers,
            num_heads,
            dropout_rate,
            attn_dropout_rate)
        
        # classifier (MLP Head is not used)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def execute(self, x):
        emb = self.embedding(x)
        emb = emb.permute(0, 2, 3, 1)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = jt.cat([cls_token, emb], dim=1)

        feat = self.encoder(emb)

        logits = self.classifier(feat[:, 0])

        return logits
    
all_acc = []

def train(model, train_loader, loss_func, optimizer, epoch, f):
    t1 = time.time()
    loss_epoch = []

    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        optimizer.step(loss)
        loss_epoch.append(loss.numpy()[0])
        data = f'Train epoch {epoch} [{(batch_idx + 1) * batch_size}/{len(train_loader)}]\tLoss: {loss.numpy()[0]}'
        print(data)
        f.write(data + '\n')
    t2 = time.time()
    data = data = f'Train epoch {epoch} average loss: {sum(loss_epoch) / len(loss_epoch)}   Time cost: {(t2-t1)/6e7}min'
    print(data)
    f.write(data + '\n')

def test(model, test_loader, epoch, f):
    model.eval()

    correct = 0
    total = 0
    accuracy = []

    for _, (inputs, labels) in tqdm(enumerate(test_loader)):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(labels.data==pred)
        correct += acc
        total += batch_size
        accuracy.append(acc / batch_size * 100)
    
    all_acc.append(sum(accuracy) / len(accuracy))
    data = f'Test epoch {epoch} accuracy: {sum(accuracy) / len(accuracy)}%'
    print(data)
    f.write(data + '\n')

if __name__ == '__main__':
    with open('Res.txt', 'w') as f:
        vit = VisionTransformer()
        loss_func = nn.CrossEntropyLoss()
        optimizer = nn.SGD(params=vit.parameters(), lr=lr, momentum=momentum)

        for epoch in range(epochs):
            train(vit, train_loader, loss_func, optimizer, epoch, f)
            test(vit, test_loader, epoch, f)
            if(all_acc[len(all_acc) - 1] >= max(all_acc)):
                vit.save('./ViT.pkl')

        print(f'Accuracy on all data in each epoch: {all_acc}')
        f.write(f'Accuracy on all data in each epoch: {all_acc}\n')
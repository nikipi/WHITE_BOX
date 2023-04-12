### known model 

class Classifier(nn.Module):
    def __init__(self, latent_dim: int, name: str = "model"):
        super(SEERClassifier, self).__init__()
        self.latent_dim = latent_dim
        self.name = name
        self.fc1 = nn.Linear(21, 4 * self.latent_dim)
        self.fc2 = nn.Linear(4 * self.latent_dim, self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 2)
        self.checkpoints_files = []
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

    def input_to_representation(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def representation_to_output(self, h):
        h = self.relu(h)
        h = self.dropout(h)
        h = self.out(h)
        return h
      
      
      
      
      
  ###### Hugging face model format 
  
  ### two key element: all_hidden_states /// mask 
  
  def get_sentence_repr(sentence, model, tokenizer, sep, model_name, device):
    """
    Get representations for one sentence
    """

    with torch.no_grad():
        ids = tokenizer.encode(sentence)
        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: list of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) (hidden_states at output of each layer plus initial embedding outputs)
        all_hidden_states = model(input_ids)[-1]
        # convert to format required for contexteval: numpy array of shape (num_layers, sequence_length, representation_dim)
        all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states]
        all_hidden_states = np.array(all_hidden_states)

    #For each word, take the representation of its last sub-word
    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert len(segmented_tokens) == all_hidden_states.shape[1], 'incompatible tokens and states'
    mask = np.full(len(segmented_tokens), False)

    if model_name.startswith('gpt2') or model_name.startswith('xlnet') or model_name.startswith('roberta'):
        # if next token is a new word, take current token's representation
        #print(segmented_tokens)
        for i in range(len(segmented_tokens)-1):
            if segmented_tokens[i+1].startswith(sep):
                #print(i)
                mask[i] = True
        # always take the last token representation for the last word
        mask[-1] = True
    # example: ['jim</w>', 'henson</w>', 'was</w>', 'a</w>', 'pup', 'pe', 'teer</w>']
    elif model_name.startswith('xlm'):
        # if current token is a new word, take it
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].endswith(sep):
                mask[i] = True
        mask[-1] = True
    elif model_name.startswith('bert') or model_name.startswith('distilbert'):
        # if next token is not a continuation, take current token's representation
        for i in range(len(segmented_tokens)-1):
            if not segmented_tokens[i+1].startswith(sep):
                mask[i] = True
        mask[-1] = True
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    all_hidden_states = all_hidden_states[:, mask]
    # all_hidden_states = torch.tensor(all_hidden_states).to(device)

    return all_hidden_states
  
 # top-level list: sentences, second-level lists: layers, third-level tensors of num_words x representation_dim
test_sentence_representations = [get_sentence_repr(sentence, model, tokenizer, sep, model_name, device) 
                                  for sentence in test_sentences]
print('test: ', len(test_sentence_representations))
### top-level: sentences 

test_sentence_representations = [list(l) for l in zip(*test_sentence_representations)]
print('test layer: ', len(test_sentence_representations))
print('sentences in each layer: ', len(test_sentence_representations[0]))

test_representations_all = [torch.tensor(np.concatenate(test_layer_representations, 0)).to(device) for test_layer_representations in test_sentence_representations]

print('test: ', len(test_representations_all[0]))
# concatenate all labels
test_labels_all = torch.tensor(np.concatenate(test_labels, 0)).to(device)


##### test_final layer

#### progressive learning 
 train_representations = train_representations_all[-1]


#### test each layer

num_layers = len(train_representations_all)

train_acc, test_acc =[], []
for l in range(num_layers):
  train_representations = train_representations_all[l]

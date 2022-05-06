import torch
import os
# reference - https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/

class SaveModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, args, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.args = args
        self.model_name = None
        
    def save_best_model(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.save_model(
                epoch, model,
                optimizer, criterion, best=True
            )
            
    def save_model(
        self, epoch, model, 
        optimizer, criterion, best=False
    ):
        mode = 'best_model' if best==True else 'final_model'
        if (mode == 'best_mode') and (self.model_name != None):
            os.remove(self.model_name)
        self.model_name = f'{self.args.exp_name}_{mode}_{epoch}.pth'
        self.path =  f'/scratch/connectome/jubin/ABCD-3DCNN-jub/suicidality/outputs/{self.args.exp_name}_{mode}_{epoch}.pth'
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, self.path)

        print(f"Save {mode} of epoch {epoch}")

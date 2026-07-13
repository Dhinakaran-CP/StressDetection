import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MultimodalTrainer:
    """
    Manages training, validation, checkpointing, and early stopping
    for both unimodal/baseline classifiers and MoE hybrid models.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, 
                 checkpoint_path='outputs/checkpoints/best_model.pt', patience=5, 
                 auxiliary_weight=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.auxiliary_weight = auxiliary_weight
        
        self.best_val_f1 = -1.0
        self.epochs_no_improve = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": []
        }
        
    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            face_x = batch['face'].to(self.device)
            voice_x = batch['voice'].to(self.device)
            physio_x = batch['physio'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Optional masks (default to None)
            f_mask = batch.get('face_mask', None)
            v_mask = batch.get('voice_mask', None)
            p_mask = batch.get('physio_mask', None)
            
            if f_mask is not None:
                f_mask = f_mask.to(self.device)
            if v_mask is not None:
                v_mask = v_mask.to(self.device)
            if p_mask is not None:
                p_mask = p_mask.to(self.device)
                
            self.optimizer.zero_grad()
            
            # Forward pass
            # Support both signature styles (baselines vs MoE)
            try:
                outputs = self.model(face_x, voice_x, physio_x, f_mask, v_mask, p_mask)
            except TypeError:
                # Fallback for baselines that don't accept masks
                outputs = self.model(face_x, voice_x, physio_x)
                
            if isinstance(outputs, tuple):
                logits, aux_loss = outputs
                loss = self.criterion(logits, labels) + self.auxiliary_weight * aux_loss
            else:
                logits = outputs
                loss = self.criterion(logits, labels)
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * face_x.size(0)
            
        return total_loss / len(self.train_loader.dataset)

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                face_x = batch['face'].to(self.device)
                voice_x = batch['voice'].to(self.device)
                physio_x = batch['physio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                f_mask = batch.get('face_mask', None)
                v_mask = batch.get('voice_mask', None)
                p_mask = batch.get('physio_mask', None)
                
                if f_mask is not None:
                    f_mask = f_mask.to(self.device)
                if v_mask is not None:
                    v_mask = v_mask.to(self.device)
                if p_mask is not None:
                    p_mask = p_mask.to(self.device)
                    
                try:
                    outputs = self.model(face_x, voice_x, physio_x, f_mask, v_mask, p_mask)
                except TypeError:
                    outputs = self.model(face_x, voice_x, physio_x)
                    
                if isinstance(outputs, tuple):
                    logits, aux_loss = outputs
                    loss = self.criterion(logits, labels) + self.auxiliary_weight * aux_loss
                else:
                    logits = outputs
                    loss = self.criterion(logits, labels)
                    
                total_loss += loss.item() * face_x.size(0)
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        # Metrics
        avg_loss = total_loss / len(self.val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        return avg_loss, acc, prec, rec, f1, cm

    def fit(self, num_epochs=50):
        """Runs the complete training and validation cycle."""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = self._validate()
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_prec"].append(val_prec)
            self.history["val_rec"].append(val_rec)
            self.history["val_f1"].append(val_f1)
            
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
            
            # Checkpoint on best Validation F1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_cm': val_cm
                }, self.checkpoint_path)
                print(f"  --> Saved new best model checkpoint to {self.checkpoint_path}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs (no improvement in F1 for {self.patience} epochs).")
                    break
                    
        return self.history

import torch
import numpy as np
from tqdm import tqdm

class Evaluator:
    """
    Run evaluation of a VLM model on dataset.
    Compatible with CLIP-style (similarity) and LLaVA-style (QA).
    """

    def __init__(self, model, dataset, batch_size=32, threshold=0.25):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.threshold = threshold

    def run(self):
        """
        Iterate through dataset in batches, compute predictions, save scores.
        Returns:
            dict with keys: scores, predictions, ground_truth
        """
        scores, preds, gts = [], [], []

        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        for batch in tqdm(loader, desc="Evaluating"):
            images, labels = batch

            # depending on model type
            if hasattr(self.model, "similarity"):  # CLIP-style
                texts = [f"a photo of a {self.dataset.classes[l]}" for l in labels]
                img_feats = self.model.encode_image(images)
                txt_feats = self.model.encode_text(texts)
                sim = self.model.similarity(img_feats, txt_feats)  # (B,B) or (B,1)
                batch_scores = sim.diag().cpu().numpy().tolist()
                scores.extend(batch_scores)
                preds.extend([1 if s >= self.threshold else 0 for s in batch_scores])

            elif hasattr(self.model, "predict"):  # LLaVA-style
                # LLaVA predict returns text answers
                prompts = [f"what is in this image?" for _ in images]
                options = [self.dataset.classes for _ in images]
                answers = self.model.predict(images, prompts, options)
                preds.extend(answers)
                # TODO: map answers back to ground truth index
                scores.extend([1.0 if a == self.dataset.classes[l] else 0.0 for a, l in zip(answers, labels)])

            gts.extend(labels.numpy().tolist())

        return {"scores": scores, "predictions": preds, "ground_truth": gts}

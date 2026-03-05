from PIL import Image
import io
import numpy as np
import torch


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def colorfulness():
    """Hasler-Susstrunk colorfulness metric."""

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        x = images.astype(np.float32)
        rg = np.abs(x[..., 0] - x[..., 1])
        yb = np.abs(0.5 * (x[..., 0] + x[..., 1]) - x[..., 2])
        std_rg = rg.std(axis=(1, 2))
        std_yb = yb.std(axis=(1, 2))
        mean_rg = rg.mean(axis=(1, 2))
        mean_yb = yb.mean(axis=(1, 2))
        scores = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        return scores.astype(np.float32), {}

    return _fn


def rms_contrast():
    """RMS contrast in grayscale (std of luminance)."""

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        x = images.astype(np.float32)
        gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        scores = gray.std(axis=(1, 2))
        return scores.astype(np.float32), {}

    return _fn


def saturation():
    """Mean HSV saturation. Higher = more vivid colors."""

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        scores = []
        for img in images:
            from PIL import Image as PILImage
            hsv = np.array(PILImage.fromarray(img).convert("HSV"), dtype=np.float32)
            scores.append(hsv[..., 1].mean() / 255.0)
        return np.array(scores, dtype=np.float32), {}

    return _fn


def entropy():
    """Pixel entropy of grayscale image. Higher = more complex/textured."""

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        scores = []
        for img in images:
            x = img.astype(np.float32)
            gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.uint8)
            counts = np.bincount(gray.ravel(), minlength=256).astype(np.float32)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            scores.append(-float(np.sum(probs * np.log2(probs))))
        return np.array(scores, dtype=np.float32), {}

    return _fn


def sharpness():
    """Laplacian variance sharpness metric.
    Higher variance of the Laplacian = sharper image (more high-frequency detail).
    """

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        x = images.astype(np.float32)
        gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

        # Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        scores = []
        for img in gray:
            from scipy.ndimage import convolve
            lap = convolve(img, kernel)
            scores.append(lap.var())
        return np.array(scores, dtype=np.float32), {}

    return _fn


def clip_iqa():
    """CLIP-IQA — CLIP-based no-reference image quality metric.
    Higher score = better perceptual quality. Uses piq.CLIPIQA.
    """
    import piq
    scorer = piq.CLIPIQA()

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            imgs = images.float().clamp(0, 1)
        else:
            imgs = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
            imgs = imgs.clamp(0, 1)
        imgs = imgs.cuda()
        scorer.to(imgs.device)
        per_image = torch.stack([scorer(imgs[i:i+1]) for i in range(len(imgs))]).squeeze()
        return per_image.detach().cpu().numpy().astype(np.float32), {}

    return _fn


def brisque():
    """BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).
    Lower BRISQUE = better quality. We negate so higher reward = better quality.
    """
    import piq

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            imgs = images.float().clamp(0, 1)
        else:
            imgs = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
            imgs = imgs.clamp(0, 1)
        imgs = imgs.cuda()
        per_image = torch.stack([piq.brisque(imgs[i:i+1], data_range=1.0) for i in range(len(imgs))])
        return -per_image.detach().cpu().numpy().astype(np.float32), {}

    return _fn


def edge_density():
    """Canny edge density — fraction of pixels identified as edges.
    Higher = more geometric structure/contours in the image.
    """
    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        from scipy.ndimage import sobel
        scores = []
        for img in images:
            x = img.astype(np.float32)
            gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
            sx = sobel(gray, axis=0)
            sy = sobel(gray, axis=1)
            magnitude = np.hypot(sx, sy)
            threshold = 0.1 * magnitude.max() if magnitude.max() > 0 else 1.0
            scores.append((magnitude > threshold).mean())
        return np.array(scores, dtype=np.float32), {}

    return _fn


def symmetry():
    """Bilateral (left-right) symmetry of luminance.
    Measured as 1 - mean absolute difference between image and its horizontal flip,
    normalised by mean luminance. Higher = more symmetric.
    """
    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        scores = []
        for img in images:
            x = img.astype(np.float32)
            gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
            flipped = np.fliplr(gray)
            mean_lum = gray.mean() + 1e-6
            score = 1.0 - np.abs(gray - flipped).mean() / mean_lum
            scores.append(float(score))
        return np.array(scores, dtype=np.float32), {}

    return _fn


def pickscore():
    """PickScore — CLIP ViT-H-14 fine-tuned on Pick-a-Pic human preference data.
    Measures human aesthetic/alignment preference. Higher = more preferred by humans.
    Model: yuvalkirstain/PickScore_v1
    """
    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").cuda()
    model.eval()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            imgs_uint8 = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu()
            pil_images = [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in imgs_uint8]
        else:
            pil_images = [Image.fromarray(img) for img in images]

        inputs = processor(
            text=list(prompts), images=pil_images,
            return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            img_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_emb = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            scores = (img_emb * txt_emb).sum(dim=-1)

        return scores.cpu().numpy().astype(np.float32), {}

    return _fn


def clip_score():
    """CLIP text-image alignment score.
    Cosine similarity between image and prompt embeddings using CLIP-ViT-B/32.
    Higher = better prompt alignment.
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            imgs_uint8 = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu()
            pil_images = [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in imgs_uint8]
        else:
            pil_images = [Image.fromarray(img) for img in images]

        inputs = processor(text=list(prompts), images=pil_images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores = (img_emb * txt_emb).sum(dim=-1)

        return scores.cpu().numpy().astype(np.float32), {}

    return _fn


def rule_of_thirds():
    """Rule-of-thirds compositional score.
    Measures how much edge energy is concentrated near the 1/3 and 2/3 gridlines
    relative to the full image, using a Gaussian weight mask.
    Higher = more salient content placed along the compositional thirds.
    """
    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        from scipy.ndimage import sobel
        scores = []
        for img in images:
            x = img.astype(np.float32)
            gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

            # Edge magnitude
            sx = sobel(gray, axis=0)
            sy = sobel(gray, axis=1)
            magnitude = np.hypot(sx, sy)

            H, W = gray.shape
            # Gaussian weight mask peaked at 1/3 and 2/3 lines (rows and cols)
            sigma = min(H, W) * 0.05  # 5% of image size
            ys = np.arange(H, dtype=np.float32)
            xs = np.arange(W, dtype=np.float32)
            yw = (np.exp(-((ys - H / 3) ** 2) / (2 * sigma ** 2)) +
                  np.exp(-((ys - 2 * H / 3) ** 2) / (2 * sigma ** 2)))
            xw = (np.exp(-((xs - W / 3) ** 2) / (2 * sigma ** 2)) +
                  np.exp(-((xs - 2 * W / 3) ** 2) / (2 * sigma ** 2)))
            weight = yw[:, None] + xw[None, :]  # additive grid mask
            weight /= weight.sum()

            weighted_energy = (magnitude * weight).sum()
            mean_energy = magnitude.mean() + 1e-6
            scores.append(float(weighted_energy / mean_energy))

        return np.array(scores, dtype=np.float32), {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn

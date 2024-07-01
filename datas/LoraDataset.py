import torch.utils.data as data
import torch
from torchvision import transforms as TF
from torchvision.transforms.functional import crop, resize, hflip

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


class LoraDataset(data.Dataset):
    def __init__(self, frames, captions=None, repeatFactor=10, resizeSize=(640, 360), randomCropSize=(352, 352), shift=0):
        super().__init__()
        self.frames = frames

        if captions is None:
            self.captions = ["an image of AnimeInterp"] * len(frames)
        else:
            # assert (len(captions) == len(frames))
            self.captions = captions

        self.repeatFactor = repeatFactor
        self.randomCropSize = randomCropSize
        self.cropX0 = resizeSize[0] - randomCropSize[0]
        self.cropY0 = resizeSize[1] - randomCropSize[1]
        self.resizeSize = resizeSize
        self.shift = shift + 1

        # Repeat frames
        self.frames = self.frames.repeat(repeatFactor, 1, 1, 1)
        self.captions = self.captions * repeatFactor

        # Generate random seeds for each repeated frame
        total_frames = len(frames) * repeatFactor
        self.seeds = [torch.randint(0, 1000, (1,)).item() for _ in range(total_frames)]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        caption = self.captions[idx]
        seed = self.seeds[idx]

        generator = torch.Generator()
        generator.manual_seed(seed)

        def randint(low=0, high=2):
            return torch.randint(low, high, (1,), generator=generator).item()

        shiftX = randint(0, self.shift) // 2 * 2
        shiftY = randint(0, self.shift) // 2 * 2
        shiftX = shiftX * -1 if randint() > 0 else shiftX
        shiftY = shiftY * -1 if randint() > 0 else shiftY

        cropX0 = randint(max(0, -shiftX), min(self.cropX0 - shiftX, self.cropX0))
        cropY0 = randint(max(0, -shiftY), min(self.cropY0, self.cropY0 - shiftY))

        cropArea = (cropX0, cropY0, cropX0 + self.randomCropSize[0], cropY0 + self.randomCropSize[1])
        cropArea = (cropX0 + shiftX // 2, cropY0 + shiftY // 2, cropX0 + shiftX // 2 + self.randomCropSize[0],
                    cropY0 + shiftY // 2 + self.randomCropSize[1])
        cropArea = (cropX0 + shiftX, cropY0 + shiftY, cropX0 + shiftX + self.randomCropSize[0],
                    cropY0 + shiftY + self.randomCropSize[1])

        def RandomResize(image):
            if randint() > 0:
                return resize(image, self.resizeSize, interpolation=TF.InterpolationMode.BICUBIC)
            return image
        def RandomCustomCrop(image):
            if randint() > 0:
                return crop(image, cropArea[1], cropArea[0], crop[3], crop[2])
            return image

        def RandomFlip(image):
            if randint() > 0:
                return hflip(image)
            return image

        trans = TF.Compose([
            TF.Lambda(RandomResize),
            TF.Lambda(RandomCustomCrop),
            TF.Lambda(RandomFlip),
        ])

        frame = trans(frame)

        return {
            "frames": frame,
            "captions": caption,
            "original_sizes": self.frames[idx].shape[1:],
            "crop_top_lefts": (cropArea[1], cropArea[0])
        }

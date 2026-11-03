"""Closed-set SapiensID evaluation: gallery/<id>.jpg vs query/<id>/**/image.jpg."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scan_dataset(root):
    root = Path(root).resolve()
    gallery_dir, query_dir = root / 'gallery', root / 'query'
    if not gallery_dir.is_dir() or not query_dir.is_dir():
        raise ValueError('Dataset must contain gallery/ and query/ directories.')
    gallery, queries = [], []
    for path in sorted(gallery_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.parent != gallery_dir:
            raise ValueError(f'Gallery images must be directly under gallery/: {path}')
        gallery.append(dict(path=path.relative_to(root).as_posix(), person_id=path.stem))
    identities = [row['person_id'] for row in gallery]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate gallery IDs: use exactly one gallery image per person.')
    for path in sorted(query_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(query_dir)
        if len(relative.parts) < 2:
            raise ValueError(f'Query image must be inside query/<person_id>/: {path}')
        queries.append(dict(path=path.relative_to(root).as_posix(), person_id=relative.parts[0]))
    if not gallery or not queries:
        raise ValueError('Gallery and query must both contain images.')
    missing = sorted({row['person_id'] for row in queries} - set(identities))
    if missing:
        raise ValueError(f'Query identities missing from gallery (exact, case-sensitive IDs): {missing}')
    return gallery, queries


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def score_one(query, gallery, scores=None, error=''):
    result = dict(query=query['path'], true_id=query['person_id'], predicted_id='',
                  predicted_gallery='', similarity='', correct=False, true_rank='',
                  reciprocal_rank=0.0, top5_correct=False, top5_ids='[]', error=error)
    if scores is None:
        return result
    if len(scores) != len(gallery) or not all(math.isfinite(float(x)) for x in scores):
        raise ValueError('Invalid similarity vector')
    # Stable tie breaking follows gallery path order; ties are not broken using labels.
    ranking = sorted(range(len(gallery)), key=lambda i: (-float(scores[i]), gallery[i]['path']))
    ids = [gallery[i]['person_id'] for i in ranking]
    first = ranking[0]
    rank = ids.index(query['person_id']) + 1
    result.update(predicted_id=ids[0], predicted_gallery=gallery[first]['path'],
                  similarity=float(scores[first]), correct=ids[0] == query['person_id'],
                  true_rank=rank, reciprocal_rank=1.0/rank, top5_correct=rank <= min(5, len(gallery)),
                  top5_ids=json.dumps(ids[:5], ensure_ascii=False))
    return result


def summarize(rows, gallery_count):
    if not rows:
        raise ValueError('No query results')
    groups = defaultdict(list)
    for row in rows:
        groups[row['true_id']].append(row)
    per_person = []
    for identity, items in sorted(groups.items()):
        per_person.append(dict(person_id=identity, queries=len(items),
                               correct=sum(r['correct'] for r in items),
                               failed=sum(bool(r['error']) for r in items),
                               top1_accuracy=sum(r['correct'] for r in items)/len(items)))
    total = len(rows)
    metrics = dict(gallery_images=gallery_count, query_images=total,
                   query_identities=len(groups), successful_queries=sum(not r['error'] for r in rows),
                   failed_queries=sum(bool(r['error']) for r in rows),
                   top1_accuracy=sum(r['correct'] for r in rows)/total,
                   top5_accuracy=sum(r['top5_correct'] for r in rows)/total,
                   effective_top5_k=min(5, gallery_count),
                   macro_top1_accuracy=sum(p['top1_accuracy'] for p in per_person)/len(per_person),
                   mAP=sum(r['reciprocal_rank'] for r in rows)/total,
                   protocol='closed-set; one gallery image per ID; failed queries score zero; no threshold')
    return metrics, per_person


class Encoder:
    def __init__(self, repo, checkpoint, yolo_weights=None):
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        task = repo / 'tasks/sapiensID'
        detector = task / 'src/aligners/keypoint_predictor/pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth'
        for required in [task / 'src', checkpoint / 'model.yaml', checkpoint / 'model.pth', detector]:
            if not required.exists():
                raise FileNotFoundError(required)
        if yolo_weights:
            import shutil
            source = Path(yolo_weights).resolve()
            target = Path.home() / '.cache/yolov8n-pose.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            if source != target:
                shutil.copyfile(source, target)
        if not (Path.home() / '.cache/yolov8n-pose.pt').is_file():
            raise FileNotFoundError('Supply --yolo-weights /path/to/yolov8n-pose.pt (see README).')
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(task))
        os.chdir(task)
        import numpy as np
        np.bool = np.bool_
        np.object = np.object_
        if not hasattr(np, 'float_'):
            np.float_ = np.float64
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('An NVIDIA GPU and CUDA-enabled PyTorch are required by this runner.')
        from general_utils.config_utils import load_config
        from src.models import get_model
        from src.aligners import get_aligner
        from src.pipelines import pipeline_from_name
        torch.manual_seed(2222)
        torch.cuda.manual_seed_all(2222)
        config = load_config(str(checkpoint / 'model.yaml'))
        # Evaluation loads the supplied complete checkpoint, not a training init path.
        config.start_from = ''
        self.model = get_model(config)
        self.model.load_state_dict_from_path(str(checkpoint / 'model.pth'))
        self.model.to('cuda').eval()
        aligner_config = load_config(str(task / 'src/aligners/configs/yolo_dfa.yaml'))
        aligner_config.rgb_mean = list(config.rgb_mean)
        aligner_config.rgb_std = list(config.rgb_std)
        aligner_config.use_cache = False
        self.aligner = get_aligner(aligner_config).to('cuda')
        self.aligner.eval()
        self.pipeline = pipeline_from_name('infer_aligner_model_pipeline', self.model, self.aligner)
        self.pipeline.integrity_check(dataset_color_space='RGB')
        self.transform = self.pipeline.make_test_transform()
        self.torch = torch

    def encode(self, path):
        from PIL import Image, ImageOps
        torch = self.torch
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            batch = self.transform(image).unsqueeze(0).to('cuda')
        with torch.inference_mode():
            vector = self.pipeline(batch).float()
            if vector.ndim != 2 or vector.shape[0] != 1 or not torch.isfinite(vector).all():
                raise RuntimeError('Invalid model embedding')
            if vector.norm().item() < 1e-12:
                raise RuntimeError('Zero embedding')
            return torch.nn.functional.normalize(vector, dim=1)[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--yolo-weights', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    for name in ['repo', 'checkpoint', 'yolo_weights']:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    root, out = args.data.resolve(), args.out.resolve()
    # Do not create results inside image folders where future scans would read them.
    for split in [root / 'gallery', root / 'query']:
        if out == split or split in out.parents:
            parser.error('--out must be outside gallery/ and query/.')
    gallery, queries = scan_dataset(root)
    out.mkdir(parents=True, exist_ok=True)
    manifest = [dict(split=split, **row) for split, items in [('gallery', gallery), ('query', queries)] for row in items]
    write_csv(out / 'manifest.csv', ['split', 'path', 'person_id'], manifest)
    hashes = {file_hash(root / row['path']) for row in gallery}
    duplicates = [row['path'] for row in queries if file_hash(root / row['path']) in hashes]
    data_info = dict(gallery_images=len(gallery), query_images=len(queries),
                     identities=len(gallery), exact_duplicate_queries=duplicates)
    (out / 'data_check.json').write_text(json.dumps(data_info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(data_info, ensure_ascii=False, indent=2), flush=True)
    if duplicates:
        print('WARNING: Some query files are byte-identical to gallery images. Accuracy may be inflated.', flush=True)
    if args.prepare_only:
        return
    if not args.repo or not args.checkpoint:
        parser.error('--repo and --checkpoint are required for model inference.')
    # A failed rerun must not leave an old metrics file looking like a new result.
    (out / 'metrics.json').unlink(missing_ok=True)
    encoder = Encoder(args.repo, args.checkpoint, args.yolo_weights)
    import numpy as np
    # Gallery failures stop the run: changing the reference set would change evaluation.
    gallery_vectors = []
    for i, row in enumerate(gallery, 1):
        print(f'Gallery [{i}/{len(gallery)}] {row["path"]}', flush=True)
        gallery_vectors.append(encoder.encode(root / row['path']))
    matrix = np.stack(gallery_vectors)
    np.save(out / 'gallery_embeddings.npy', matrix)
    results, query_vectors = [], []
    fields = ['query', 'true_id', 'predicted_id', 'predicted_gallery', 'similarity', 'correct',
              'true_rank', 'reciprocal_rank', 'top5_correct', 'top5_ids', 'error']
    # Write each prediction immediately so a hardware interruption preserves progress.
    with (out / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(queries, 1):
            try:
                vector = encoder.encode(root / row['path'])
                result = score_one(row, gallery, matrix @ vector)
                query_vectors.append(vector)
            except encoder.torch.cuda.OutOfMemoryError:
                raise  # A configuration error must not silently mark all remaining images wrong.
            except Exception as exc:
                result = score_one(row, gallery, error=f'{type(exc).__name__}: {exc}')
                query_vectors.append(np.full(matrix.shape[1], np.nan, dtype=np.float32))
            results.append(result)
            writer.writerow(result)
            stream.flush()
            print(f'Query [{i}/{len(queries)}] {row["path"]} -> {result["predicted_id"] or "ERROR"}', flush=True)
    np.save(out / 'query_embeddings.npy', np.stack(query_vectors))
    metrics, per_person = summarize(results, len(gallery))
    metrics.update(exact_duplicate_query_count=len(duplicates), repo=str(args.repo.resolve()),
                   checkpoint=str(args.checkpoint.resolve()), data=str(root))
    write_csv(out / 'per_person.csv', ['person_id', 'queries', 'correct', 'failed', 'top1_accuracy'], per_person)
    write_csv(out / 'wrong_predictions.csv', fields, [row for row in results if not row['correct']])
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()

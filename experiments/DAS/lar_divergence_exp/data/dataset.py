"""
NLM Chest X-Ray dataset loader.

Pairs PNG images with XML radiology reports by CXR ID.
Extracts structured auto-tags to identify contradiction cases
(Stream A finding present, Stream B report says absent — or vice versa).

Expected layout after extraction:
    data/images/    ← PNGs from NLMCXR_png.tgz
    data/reports/   ← XMLs from ecgen-radiology.tar.gz
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class CXRSample:
    cxr_id: str
    image_path: Path
    findings: str
    impression: str
    # Structured auto-tags: finding name → True/False
    tags: dict = field(default_factory=dict)
    # Set to True if image tags and report text contradict each other
    is_contradiction: bool = False


def parse_report_xml(xml_path: Path) -> dict:
    """Parse a single NLM radiology report XML into structured fields."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    findings = ""
    impression = ""
    tags = {}

    # Extract findings and impression text
    for abstract_text in root.iter("AbstractText"):
        label = abstract_text.get("Label", "").upper()
        text = (abstract_text.text or "").strip()
        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text

    # Extract MeSH major terms — these are the structured image finding labels
    # Format: <MeSH><major>cardiomegaly/mild</major></MeSH>
    # Store the root finding (before the first slash qualifier)
    for mesh in root.iter("MeSH"):
        for major in mesh.iter("major"):
            tag_text = (major.text or "").strip().lower()
            if tag_text and tag_text not in ("normal", "no indexing"):
                root_term = tag_text.split("/")[0].strip()
                tags[root_term] = True
                tags[tag_text] = True  # also store full term

    return {"findings": findings, "impression": impression, "tags": tags}


def load_dataset(
    images_dir: str,
    reports_dir: str,
) -> list[CXRSample]:
    """
    Load and pair all images with their reports.
    Returns a list of CXRSample objects.
    """
    images_dir = Path(images_dir)
    reports_dir = Path(reports_dir)

    # Map CXR ID → image paths (one patient may have frontal + lateral)
    image_map: dict[str, list[Path]] = {}
    for png in images_dir.glob("*.png"):
        # Filename format: CXR{id}_IM-{study}-{view}.png
        cxr_id = png.name.split("_")[0]  # e.g. CXR3781
        image_map.setdefault(cxr_id, []).append(png)

    samples = []
    for xml_path in reports_dir.glob("*.xml"):
        cxr_id = "CXR" + xml_path.stem  # e.g. 3781.xml → CXR3781
        if cxr_id not in image_map:
            continue

        report = parse_report_xml(xml_path)
        if not report["findings"] and not report["impression"]:
            continue

        # Prefer frontal view (view code 1001) over lateral
        images = sorted(image_map[cxr_id])
        frontal = next(
            (p for p in images if "1001" in p.name or "2001" in p.name),
            images[0],
        )

        samples.append(CXRSample(
            cxr_id=cxr_id,
            image_path=frontal,
            findings=report["findings"],
            impression=report["impression"],
            tags=report["tags"],
        ))

    print(f"Loaded {len(samples)} paired image-report samples.")
    return samples


def build_contradiction_subset(samples: list[CXRSample]) -> tuple[list, list]:
    """
    Identify contradiction cases: structured tag present in image labels
    but the finding is explicitly negated in the report text, or vice versa.

    Returns:
        normal_cases: samples where image and report agree
        contradiction_cases: samples where they structurally disagree
    """
    # Negation patterns in radiology reports
    negation_phrases = [
        "no ", "without ", "absent", "negative", "clear", "unremarkable",
        "normal", "not seen", "not identified", "no evidence of",
    ]

    # MeSH root term → report text search terms (MeSH uses different terminology)
    target_findings = {
        "cardiomegaly":       ["cardiomegaly"],
        "pulmonary atelectasis": ["atelectasis"],
        "pleural effusion":   ["pleural effusion", "effusion"],
        "opacity":            ["opacity", "opacit"],
        "pneumothorax":       ["pneumothorax"],
        "consolidation":      ["consolidation"],
        "pulmonary edema":    ["edema", "pulmonary edema"],
    }

    normal_cases = []
    contradiction_cases = []

    for sample in samples:
        report_lower = (sample.findings + " " + sample.impression).lower()
        contradicted = False

        for mesh_root, report_terms in target_findings.items():
            # Check if MeSH tag says this finding is present in the image
            if mesh_root not in sample.tags:
                continue
            # Image says finding is present — check if report explicitly negates it
            finding_mentioned = any(term in report_lower for term in report_terms)
            report_negated = any(
                f"{neg}{term}" in report_lower
                or f"{neg}evidence of {term}" in report_lower
                for neg in negation_phrases
                for term in report_terms
            )
            if report_negated or not finding_mentioned:
                contradicted = True
                break

        if contradicted:
            sample.is_contradiction = True
            contradiction_cases.append(sample)
        else:
            normal_cases.append(sample)

    print(f"Normal cases:        {len(normal_cases)}")
    print(f"Contradiction cases: {len(contradiction_cases)}")
    return normal_cases, contradiction_cases


def to_dataframe(samples: list[CXRSample]) -> pd.DataFrame:
    return pd.DataFrame([{
        "cxr_id": s.cxr_id,
        "image_path": str(s.image_path),
        "findings": s.findings,
        "impression": s.impression,
        "tags": str(s.tags),
        "is_contradiction": s.is_contradiction,
    } for s in samples])


class CXRTorchDataset(Dataset):
    """PyTorch Dataset wrapping CXRSample list, for use with DataLoader."""

    def __init__(self, samples: list[CXRSample], image_transform=None):
        self.samples = samples
        self.image_transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        return {
            "cxr_id": sample.cxr_id,
            "image": image,
            "report": sample.findings + " " + sample.impression,
            "is_contradiction": int(sample.is_contradiction),
        }

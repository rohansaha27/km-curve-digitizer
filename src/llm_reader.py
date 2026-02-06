"""
LLM-based plot metadata reader.

Uses Claude's vision capabilities to extract structured metadata from KM plots:
- Axis ranges and labels
- Number and colors of curves
- Legend information
- Plot bounding box estimation
- Direct survival probability estimates (for validation/fallback)
"""

import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def encode_image(image_path: str) -> str:
    """Encode image as base64 for API calls."""
    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


def get_image_media_type(image_path: str) -> str:
    """Get MIME type for an image file."""
    suffix = Path(image_path).suffix.lower()
    return {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }.get(suffix, 'image/png')


def extract_plot_metadata(
    image_path: str,
    client=None,
    model: str = 'claude-sonnet-4-20250514',
) -> Dict:
    """Extract structured metadata from a KM plot image using Claude.

    Args:
        image_path: Path to the KM plot image.
        client: Anthropic client instance. If None, creates one.
        model: Claude model to use.

    Returns:
        Dictionary with plot metadata including axis ranges, curve info, etc.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    image_data = encode_image(image_path)
    media_type = get_image_media_type(image_path)

    prompt = """Analyze this Kaplan-Meier survival curve plot carefully.

Extract the following information and return it as a JSON object:

{
  "x_axis": {
    "label": "the x-axis label text",
    "min": <minimum value on x-axis, as number>,
    "max": <maximum value on x-axis, as number>,
    "tick_values": [<list of visible tick mark values>]
  },
  "y_axis": {
    "label": "the y-axis label text",
    "min": <minimum value on y-axis, as number>,
    "max": <maximum value on y-axis, as number>,
    "tick_values": [<list of visible tick mark values>]
  },
  "curves": [
    {
      "label": "curve label from legend or description",
      "color_description": "description of the color (e.g., 'blue', 'red')",
      "color_hex": "#RRGGBB approximate hex color",
      "line_style": "solid, dashed, or dotted"
    }
  ],
  "has_confidence_intervals": true/false,
  "has_censoring_marks": true/false,
  "has_grid_lines": true/false,
  "has_number_at_risk": true/false,
  "title": "plot title if any, or empty string",
  "plot_region": {
    "description": "Estimate the plot area boundaries as fractions of the total image dimensions (0-1 scale). left=0 is the left edge, top=0 is the top edge.",
    "left_frac": <fraction from left edge where plot area starts>,
    "right_frac": <fraction from left edge where plot area ends>,
    "top_frac": <fraction from top edge where plot area starts>,
    "bottom_frac": <fraction from top edge where plot area ends>
  }
}

Be precise with the axis values. Look at the tick marks carefully.
For hex colors, give your best approximation of the actual curve colors.
Return ONLY the JSON object, no other text."""

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': media_type,
                        'data': image_data,
                    }
                },
                {
                    'type': 'text',
                    'text': prompt,
                }
            ]
        }]
    )

    # Parse the JSON response
    response_text = response.content[0].text.strip()
    # Handle markdown code blocks
    if response_text.startswith('```'):
        response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
        response_text = re.sub(r'\n?```$', '', response_text)

    try:
        metadata = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            metadata = json.loads(json_match.group())
        else:
            raise ValueError(f"Could not parse LLM response as JSON: {response_text[:200]}")

    return metadata


def read_survival_values(
    image_path: str,
    time_points: List[float],
    client=None,
    model: str = 'claude-sonnet-4-20250514',
) -> Dict[str, List[Dict[str, float]]]:
    """Ask the LLM to directly read survival probabilities at given time points.

    This provides a fallback/validation for the CV-based extraction.

    Args:
        image_path: Path to the KM plot image.
        time_points: List of time values to read survival at.
        client: Anthropic client instance.
        model: Claude model to use.

    Returns:
        Dictionary mapping curve labels to lists of {time, survival} dicts.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    image_data = encode_image(image_path)
    media_type = get_image_media_type(image_path)

    time_str = ', '.join([str(t) for t in time_points])

    prompt = f"""Look at this Kaplan-Meier survival curve plot very carefully.

For EACH curve in the plot, estimate the survival probability (y-value) at these time points: [{time_str}]

Read the values as precisely as possible by looking at where each curve intersects vertical lines at each time point.

Return ONLY a JSON object in this format:
{{
  "curves": [
    {{
      "label": "curve name",
      "values": [
        {{"time": <time>, "survival": <survival probability between 0 and 1>}},
        ...
      ]
    }}
  ]
}}

If a curve has ended (dropped to 0 or is no longer visible) before a time point, use the last known value or 0.
Be as precise as possible - try to estimate to 2 decimal places.
Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': media_type,
                        'data': image_data,
                    }
                },
                {
                    'type': 'text',
                    'text': prompt,
                }
            ]
        }]
    )

    response_text = response.content[0].text.strip()
    if response_text.startswith('```'):
        response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
        response_text = re.sub(r'\n?```$', '', response_text)

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError(f"Could not parse LLM response: {response_text[:200]}")

    return result


def metadata_to_extraction_params(
    metadata: Dict,
    image_width: int,
    image_height: int,
) -> Dict:
    """Convert LLM metadata to parameters for CV extraction.

    Args:
        metadata: Output from extract_plot_metadata.
        image_width, image_height: Image dimensions in pixels.

    Returns:
        Dictionary with parameters for the CV extractor.
    """
    plot_region = metadata.get('plot_region', {})

    # Convert fractional coordinates to pixel coordinates
    plot_bbox = {
        'x_min': plot_region.get('left_frac', 0.12) * image_width,
        'x_max': plot_region.get('right_frac', 0.95) * image_width,
        'y_min': plot_region.get('top_frac', 0.05) * image_height,
        'y_max': plot_region.get('bottom_frac', 0.85) * image_height,
    }

    x_axis = metadata.get('x_axis', {})
    y_axis = metadata.get('y_axis', {})

    x_range = (
        x_axis.get('min', 0),
        x_axis.get('max', 100),
    )
    y_range = (
        y_axis.get('min', 0),
        y_axis.get('max', 1.0),
    )

    curves_info = []
    for curve in metadata.get('curves', []):
        curves_info.append({
            'label': curve.get('label', 'Unknown'),
            'color_hex': curve.get('color_hex', '#000000'),
            'color_description': curve.get('color_description', ''),
            'line_style': curve.get('line_style', 'solid'),
        })

    return {
        'plot_bbox': plot_bbox,
        'x_range': x_range,
        'y_range': y_range,
        'curves': curves_info,
        'has_ci': metadata.get('has_confidence_intervals', False),
        'has_censoring': metadata.get('has_censoring_marks', False),
        'has_grid': metadata.get('has_grid_lines', False),
    }

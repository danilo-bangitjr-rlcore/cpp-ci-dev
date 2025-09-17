import React from 'react';
import { NumberLine } from './NumberLine';
import { RangeInput } from './RangeInput';
import { ConfigDetails } from './ConfigDetails';
import type { ObsCardConfig, ObsTagConfig } from '../../types/tag-types';

type ObsCardProps = {
  config: ObsCardConfig;
  onConfigChange: (config: ObsCardConfig) => void;
  onSave?: () => void;
};

const TICK_CONFIG = [
  { label: 'Eng Min', color: '#2196f3' },
  { label: 'Lo Red', color: '#f44336' },
  { label: 'Lo Yellow', color: '#ffc107' },
  { label: 'Op Min', color: '#4caf50' },
  { label: 'Op Max', color: '#4caf50' },
  { label: 'Hi Yellow', color: '#ffc107' },
  { label: 'Hi Red', color: '#f44336' },
  { label: 'Eng Max', color: '#2196f3' },
];

function tagToValues(tag: ObsTagConfig): number[] {
  return [
    tag.operating_range?.[0] ?? 0,
    tag.red_bounds?.[0] ?? 0,
    tag.yellow_bounds?.[0] ?? 0,
    tag.expected_range?.[0] ?? 0,
    tag.expected_range?.[1] ?? 0,
    tag.yellow_bounds?.[1] ?? 0,
    tag.red_bounds?.[1] ?? 0,
    tag.operating_range?.[1] ?? 0,
  ];
}

function valuesToTag(values: number[]): ObsTagConfig {
  return {
    operating_range: [values[0], values[7]],
    red_bounds: [values[1], values[6]],
    yellow_bounds: [values[2], values[5]],
    expected_range: [values[3], values[4]],
  };
}

const validateRange = (vals: number[]) => {
  const [engMin, loRed, loYellow, opMin, opMax, hiYellow, hiRed, engMax] = vals;
  if (
    vals.length !== 8 ||
    vals.some((v) => typeof v !== 'number' || isNaN(v))
  ) {
    return false;
  }
  if (
    engMin > engMax ||
    loRed > hiRed ||
    loYellow > hiYellow ||
    opMin > opMax
  ) {
    return false;
  }
  return true;
};

function getMinMax(tags: ObsTagConfig[]): [number, number] {
  const values = tags.flatMap(tagToValues);
  if (values.length === 0) return [-10, 10];
  const min = Math.min(...values) - 10;
  const max = Math.max(...values) + 10;
  return [min, max];
}

export const ObsCard: React.FC<ObsCardProps> = ({
  config,
  onConfigChange,
  onSave,
}) => {
  const handleTagChange = (idx: number, values: number[]) => {
    const updatedTags = config.tags.map((tag, i) =>
      i === idx ? valuesToTag(values) : tag
    );
    onConfigChange({ ...config, tags: updatedTags });
  };

  const handleNameChange = (name: string) => {
    onConfigChange({ ...config, name });
  };

  const handleConnectionIdChange = (connection_id: string) => {
    onConfigChange({ ...config, connection_id });
  };

  const [min, max] = getMinMax(config.tags);

  return (
    <div className="flex-1 basis-[calc(50%-1rem)] border border-gray-300 bg-gray-100 p-4 rounded-lg shadow-sm min-h-[200px] flex flex-col justify-between justify-center">
      <div className="flex flex-row gap-8">
        <ConfigDetails
          name={config.name}
          onNameChange={handleNameChange}
          connection_id={config.connection_id || ''}
          onConnectionIdChange={handleConnectionIdChange}
        />
        <div className="flex-1">
          <div className="my-4">
            <NumberLine
              min={min}
              max={max}
              ranges={config.tags.map((tag) => ({
                label: config.name,
                values: tagToValues(tag),
              }))}
              tickConfig={TICK_CONFIG}
              isValid={config.tags.every((tag) =>
                validateRange(tagToValues(tag))
              )}
            />
          </div>
          <div className="flex gap-10 mb-4 justify-center text-center">
            {config.tags.map((tag, i) => (
              <RangeInput
                key={config.name}
                value={tagToValues(tag)}
                onChange={(v) => handleTagChange(i, v)}
                valid={validateRange(tagToValues(tag))}
                tickConfig={TICK_CONFIG}
              />
            ))}
          </div>
        </div>
      </div>
      {config.modified && onSave && (
        <button
          onClick={onSave}
          className="mt-4 px-4 py-2 rounded bg-blue-500 text-white hover:bg-blue-600 transition"
          type="button"
        >
          Save
        </button>
      )}
    </div>
  );
};

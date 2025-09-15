import React, { useState } from 'react';
import { NumberLine } from './NumberLine';
import { RangeInput } from './RangeInput';
import { ConfigDetails } from './ConfigDetails';

type TagConfig = {
  label: string;
  values: [number, number];
};

type ObsCardConfig = {
  name: string;
  tags: TagConfig[];
  dataType?: string;
  connection_id?: string;
};

type ObsCardProps = {
  config: ObsCardConfig;
};

const COLORS = [
  '#4caf50',
  '#2196f3',
  '#ff9800',
  '#e91e63',
  '#9c27b0',
  '#f44336',
  '#00bcd4',
  '#8bc34a',
  '#ffc107',
  '#795548',
];

const validateRange = ([low, high]: [number, number]) => low <= high;

function getMinMax(tags: TagConfig[]): [number, number] {
  const values = tags.flatMap((t) => t.values);
  if (values.length === 0) return [-10, 10];
  const min = Math.min(...values) - 10;
  const max = Math.max(...values) + 10;
  return [min, max];
}

function getColor(idx: number): string {
  return COLORS[idx % COLORS.length];
}

export const ObsCard: React.FC<ObsCardProps> = ({ config }) => {
  const [tags, setTags] = useState<TagConfig[]>(config.tags);
  const [name, setName] = useState(config.name);
  const [connection_id, setConnectionId] = useState(config.connection_id || '');

  const handleTagChange = (idx: number, values: [number, number]) => {
    setTags((tags) => tags.map((t, i) => (i === idx ? { ...t, values } : t)));
  };

  const [min, max] = getMinMax(tags);

  return (
    <div className="flex-1 basis-[calc(50%-1rem)] border border-gray-300 bg-gray-100 p-4 rounded-lg shadow-sm min-h-[200px] flex flex-col justify-between max-w-3xl justify-center">
      <div className="flex flex-row gap-8">
        <ConfigDetails
          name={name}
          onNameChange={setName}
          connection_id={connection_id}
          onConnectionIdChange={setConnectionId}
        />
        <div className="flex-1 space-y-2">
          <div className="my-4">
            <NumberLine
              min={min}
              max={max}
              ranges={tags.map((t, i) => ({
                label: t.label,
                values: t.values,
                color: validateRange(t.values) ? getColor(i) : 'red',
              }))}
            />
          </div>
          <div className="flex gap-10 mb-4 justify-center">
            {tags.map((t, i) => (
              <RangeInput
                key={t.label}
                label={t.label}
                color={getColor(i)}
                value={t.values}
                onChange={(v) => handleTagChange(i, v)}
                valid={validateRange(t.values)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

import React from 'react';

type TickConfig = {
  label: string;
  color: string;
};

type RangeInputProps = {
  value: number[];
  onChange: (value: number[]) => void;
  valid: boolean;
  tickConfig: TickConfig[];
};

export const RangeInput: React.FC<RangeInputProps> = ({
  value,
  onChange,
  valid,
  tickConfig,
}) => {
  const handleChange = (idx: number, v: number) => {
    const next = value.slice();
    next[idx] = v;
    onChange(next);
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {tickConfig.map((tick, idx) => (
          <div
            key={tick.label}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <input
              type="number"
              step="any"
              value={value[idx] ?? ''}
              style={{
                border: `1px solid ${tick.color}`,
                maxWidth: 60,
                textAlign: 'center',
              }}
              onChange={(e) =>
                handleChange(idx, parseFloat(e.target.value) || 0)
              }
            />
            <span style={{ fontSize: 11, color: tick.color }}>
              {tick.label}
            </span>
          </div>
        ))}
      </div>
      {!valid && (
        <span style={{ color: 'red', fontSize: 12 }}>Invalid range values</span>
      )}
    </div>
  );
};

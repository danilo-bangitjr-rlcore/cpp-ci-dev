import React from 'react';

type Range = {
  label: string;
  values: number[];
};

type TickConfig = {
  label: string;
  color: string;
};

type NumberLineProps = {
  ranges: Range[];
  min?: number;
  max?: number;
  tickConfig: TickConfig[];
  isValid: boolean;
};

export const NumberLine: React.FC<NumberLineProps> = ({
  ranges,
  min = 0,
  max = 100,
  tickConfig,
  isValid,
}) => {
  const toPercent = (value: number) => ((value - min) / (max - min)) * 100;

  return (
    <div style={{ marginTop: 20 }}>
      <div
        style={{
          position: 'relative',
          height: 50,
          background: '#ddd',
          borderRadius: 3,
        }}
      >
        {ranges.map(({ values, label }) =>
          values.map((val, idx) => {
            const percent = toPercent(val);
            const tick = tickConfig[idx] || { label: 'Unknown', color: '#999' };
            const color = isValid ? tick.color : '#f44336';
            return (
              <React.Fragment key={`${label}-${idx}`}>
                <div
                  style={{
                    position: 'absolute',
                    left: `${percent}%`,
                    top: -6,
                    width: 3,
                    height: 56,
                    background: color,
                    border: !isValid ? '1px dashed #f44336' : undefined,
                    borderRadius: 2,
                  }}
                />
                <div
                  style={{
                    position: 'absolute',
                    left: `calc(${percent}% - 5px)`,
                    top: 50,
                    width: 12,
                    height: 12,
                    background: color,
                    borderRadius: '50%',
                    border: '2px solid #fff',
                    boxShadow: '0 0 2px #888',
                  }}
                />
              </React.Fragment>
            );
          })
        )}
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 15,
          marginTop: 4,
        }}
      >
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
};

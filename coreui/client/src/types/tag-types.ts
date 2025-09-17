export type RawTag = {
  name?: string;
  connection_id?: string;
  operating_range?: [number, number];
  expected_range?: [number, number];
  yellow_bounds?: [number, number];
  red_bounds?: [number, number];
  is_computed?: boolean;
};

export type ObsTagConfig = {
  operating_range?: [number, number];
  expected_range?: [number, number];
  yellow_bounds?: [number, number];
  red_bounds?: [number, number];
};

export type ObsCardConfig = {
  name: string;
  tags: ObsTagConfig[];
  dataType?: string;
  connection_id?: string;
  modified?: boolean;
  originalIndex: number;
};

export type TagResponse = {
  message: string;
  tag: { name: string };
  index: number;
};

export function buildObsTagConfig(tag: RawTag): ObsTagConfig {
  return {
    operating_range: tag.operating_range,
    expected_range: tag.expected_range,
    yellow_bounds: tag.yellow_bounds,
    red_bounds: tag.red_bounds,
  };
}

export function buildObsConfig(tag: RawTag, index: number): ObsCardConfig {
  return {
    name: tag.name ?? '',
    connection_id: tag.connection_id,
    tags: [buildObsTagConfig(tag)],
    modified: false,
    originalIndex: index,
  };
}

export function createEmptyObsConfig(index: number): ObsCardConfig {
  return {
    name: '',
    connection_id: '',
    tags: [
      {
        operating_range: [0, 0],
        expected_range: [0, 0],
        yellow_bounds: [0, 0],
        red_bounds: [0, 0],
      },
    ],
    modified: true,
    originalIndex: index,
  };
}

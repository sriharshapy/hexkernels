Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out, int n,
                          const int8_t *tableA, const int8_t *tableB);
Apply a DUAL-TABLE piecewise LUT selected by the SIGN of each input:
  if in[i] < 0:  out[i] = tableA[(uint8_t)in[i] - 128]
                 (maps -128->tableA[0], -1->tableA[127])
  if in[i] >= 0: out[i] = tableB[in[i]]
                 (maps 0->tableB[0], 127->tableB[127])
tableA has 128 entries (negative half). tableB has 128 entries (non-negative half).
Both tables are supplied as runtime pointers — do NOT hardcode their values.
The sign test is on the SIGNED value in[i], not on the index.
n=1000 (not a multiple of 128 — handle the tail).
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.

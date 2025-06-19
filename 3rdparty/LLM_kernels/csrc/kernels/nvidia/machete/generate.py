# Copyright 2025 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from
# [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/generate.py

import itertools
import math
import os
import shutil
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import reduce
from typing import Optional, Union
import argparse

import jinja2
# yapf conflicts with isort for this block
# yapf: disable
import enum
from typing import Union

from cutlass_library.library import enum_auto, DataTypeNames, DataType, DataTypeTag, DataTypeSize, KernelScheduleTag, KernelScheduleType, EpilogueScheduleTag, EpilogueScheduleType, TileSchedulerType, TileSchedulerTag

class KLLMDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecialized = enum_auto()
    TmaWarpSpecializedPingpong = enum_auto()
    TmaWarpSpecializedCooperative = enum_auto()


KLLMDataTypeNames: dict[Union[KLLMDataType, DataType], str] = {
    **DataTypeNames,  # type: ignore
    **{
        KLLMDataType.u4b8: "u4b8",
        KLLMDataType.u8b128: "u8b128",
    }
}

KLLMDataTypeTag: dict[Union[KLLMDataType, DataType], str] = {
    **DataTypeTag,  # type: ignore
    **{
        KLLMDataType.u4b8: "cutlass::vllm_uint4b8_t",
        KLLMDataType.u8b128: "cutlass::vllm_uint8b128_t",
    }
}

KLLMDataTypeSize: dict[Union[KLLMDataType, DataType], int] = {
    **DataTypeSize,  # type: ignore
    **{
        KLLMDataType.u4b8: 4,
        KLLMDataType.u8b128: 8,
    }
}

KLLMDataTypeKLLMScalarTypeTag: dict[Union[KLLMDataType, DataType], str] = {
    KLLMDataType.u4b8: "llm_kernels::nvidia::vllm_dtype::kU4B8",
    KLLMDataType.u8b128: "llm_kernels::nvidia::vllm_dtype::kU8B128",
    DataType.u4: "llm_kernels::nvidia::vllm_dtype::kU4",
    DataType.u8: "llm_kernels::nvidia::vllm_dtype::kU8",
    DataType.s4: "llm_kernels::nvidia::vllm_dtype::kS4",
    DataType.s8: "llm_kernels::nvidia::vllm_dtype::kS8",
    DataType.f16: "llm_kernels::nvidia::vllm_dtype::kHalf",
    DataType.bf16: "llm_kernels::nvidia::vllm_dtype::kBFloat16",
}

KLLMDataTypeTorchDataTypeTag: dict[Union[KLLMDataType, DataType], str] = {
    # DataType.u8: "at::ScalarType::Byte",
    DataType.s8: "llm_kernels::nvidia::vllm_dtype::kInt8",
    DataType.e4m3: "llm_kernels::nvidia::vllm_dtype::kFloat8_e4m3fn",
    # DataType.s32: "at::ScalarType::Int",
    DataType.f16: "llm_kernels::nvidia::vllm_dtype::kHalf",
    DataType.bf16: "llm_kernels::nvidia::vllm_dtype::kBFloat16",
    DataType.f32: "llm_kernels::nvidia::vllm_dtype::kFloat",
}

KLLMKernelScheduleTag: dict[Union[
    MixedInputKernelScheduleType, KernelScheduleType], str] = {
        **KernelScheduleTag,  # type: ignore
        **{
            MixedInputKernelScheduleType.TmaWarpSpecialized:
            "cutlass::gemm::KernelTmaWarpSpecialized",
            MixedInputKernelScheduleType.TmaWarpSpecializedPingpong:
            "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
            MixedInputKernelScheduleType.TmaWarpSpecializedCooperative:
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
        }
    }



# yapf: enable

#
#   Generator templating
#

DISPATCH_TEMPLATE = """
/*
 * Copyright 2025 vLLM Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/generate.py
 */

#include "csrc/kernels/nvidia/machete/machete_mm_launcher.cuh"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace machete {

#if defined(ENABLE_MACHETE)

{% for impl_config in impl_configs %}
{% set type_sig = gen_type_sig(impl_config.types) -%}
{% for s in impl_config.schedules %}
extern void impl_{{type_sig}}_sch_{{gen_sch_sig(s)}}(MMArgs);
{%- endfor %}

void mm_dispatch_{{type_sig}}(MMArgs args) {
  auto M = args.M;
  auto N = args.N;
  auto K = args.K;
    
  if (!args.maybe_schedule) {
    {%- for cond, s in impl_config.heuristic %}
    {%if cond is not none%}if ({{cond}})
    {%- else %}else
    {%- endif %}
        return impl_{{type_sig}}_sch_{{ gen_sch_sig(s) }}(args);{% endfor %}
  }

  {%- for s in impl_config.schedules %}
  if (*args.maybe_schedule == "{{ gen_sch_sig(s) }}")
    return impl_{{type_sig}}_sch_{{ gen_sch_sig(s) }}(args);
  {%- endfor %}
  KLLM_KERNEL_CHECK_WITH_INFO(false, fmtstr("machete_gemm(..) is not implemented for "
                                     "schedule = {}", *args.maybe_schedule));
}
{%- endfor %}


static inline std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_scalartype(
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& t) {
    if (!t) {
      return std::nullopt;
    } else {
      return *t;
    };
}

void mm_dispatch(MMArgs args) {
  auto out_type = args.maybe_out_type.value_or(args.a_type);
  auto a_type = args.a_type;
  auto maybe_g_scales_type = maybe_scalartype(args.maybe_group_scales_type);
  auto maybe_g_zeros_type = maybe_scalartype(args.maybe_group_zeros_type);
  auto maybe_ch_scales_type = maybe_scalartype(args.maybe_channel_scales_type);
  auto maybe_tok_scales_type = maybe_scalartype(args.maybe_token_scales_type);

  {% for impl_config in impl_configs %}
  {% set t = impl_config.types -%}
  {% set type_sig = gen_type_sig(t) -%}
  if (args.b_type == {{KLLMScalarTypeTag[t.b]}}
      && a_type == {{TorchTypeTag[t.a]}}
      && out_type == {{TorchTypeTag[t.out]}}
      && {%if t.b_group_scale != void -%}
      maybe_g_scales_type == {{TorchTypeTag[t.b_group_scale]}}
      {%- else %}!maybe_g_scales_type{%endif%}
      && {%if t.b_group_zeropoint != void -%}
      maybe_g_zeros_type == {{TorchTypeTag[t.b_group_zeropoint]}}
      {%- else %}!maybe_g_zeros_type{%endif%}
      && {%if t.b_channel_scale != void -%}
      maybe_ch_scales_type == {{TorchTypeTag[t.b_channel_scale]}}
      {%- else %}!maybe_ch_scales_type{%endif%}
      && {%if t.a_token_scale != void -%}
      maybe_tok_scales_type == {{TorchTypeTag[t.a_token_scale]}}
      {%- else %}!maybe_tok_scales_type{%endif%}
  ) {
      return mm_dispatch_{{type_sig}}(args);
  }
  {%- endfor %}
  
  KLLM_KERNEL_CHECK_WITH_INFO(
    false, fmtstr("machete_mm(..) is not implemented for a_type={}, b_type={}, out_type={}, with_group_scale_type={}, with_group_zeropoint_type={}, with_channel_scale_type={}, with_token_scale_type={}", args.a_type.str(), args.b_type.str(), out_type, maybe_g_scales_type ? (*maybe_g_scales_type).str() : "None", maybe_g_zeros_type ? (*maybe_g_zeros_type).str() : "None", maybe_ch_scales_type ? (*maybe_ch_scales_type).str() : "None", maybe_tok_scales_type ? (*maybe_tok_scales_type).str() : "None"));
}

std::vector<std::string> supported_schedules_dispatch(
    SupportedSchedulesArgs args) {
    auto out_type = args.maybe_out_type.value_or(args.a_type);
    
    {% for impl_config in impl_configs %}
    {% set t = impl_config.types -%}
    {% set schs = impl_config.schedules -%}
    if (args.b_type == {{KLLMScalarTypeTag[t.b]}}
        && args.a_type == {{TorchTypeTag[t.a]}}
        && out_type == {{TorchTypeTag[t.out]}}
        && {%if t.b_group_scale != void -%}
        args.maybe_group_scales_type == {{TorchTypeTag[t.b_group_scale]}}
        {%- else %}!args.maybe_group_scales_type{%endif%}
        && {%if t.b_group_zeropoint != void-%}
        args.maybe_group_zeros_type == {{TorchTypeTag[t.b_group_zeropoint]}}
        {%- else %}!args.maybe_group_zeros_type{%endif%}
    ) {
        return {
            {%- for s in impl_config.schedules %}
            "{{gen_sch_sig(s)}}"{% if not loop.last %},{% endif %}
            {%- endfor %}
        };
    }
    {%- endfor %}
    
    return {};
};

#endif

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels
"""

IMPL_TEMPLATE = """
/*
 * Copyright 2025 vLLM Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/generate.py
 */

#include "csrc/kernels/nvidia/machete/machete_mm_launcher.cuh"

namespace llm_kernels {
namespace nvidia {
namespace machete {

#if defined(ENABLE_MACHETE)

{% for sch in unique_schedules(impl_configs) %}
{% set sch_sig = gen_sch_sig(sch) -%}
struct sch_{{sch_sig}} {
  using TileShapeNM = Shape<{{
      to_cute_constant(sch.tile_shape_mn)|join(', ')}}>;
  using ClusterShape = Shape<{{
      to_cute_constant(sch.cluster_shape_mnk)|join(', ')}}>;
  // TODO: Reimplement
  // using KernelSchedule   = {{KernelScheduleTag[sch.kernel_schedule]}};
  using EpilogueSchedule = {{EpilogueScheduleTag[sch.epilogue_schedule]}};
  using TileScheduler    = {{TileSchedulerTag[sch.tile_scheduler]}};
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};
{% endfor %}
    
{% for impl_config in impl_configs %}
{% set t = impl_config.types -%}
{% set schs = impl_config.schedules -%}
{% set type_sig = gen_type_sig(t) -%}

template<typename Sch>
using Kernel_{{type_sig}} = MacheteKernelTemplate<
  {{DataTypeTag[t.a]}},  // ElementA
  {{DataTypeTag[t.b]}},  // ElementB
  {{DataTypeTag[t.out]}},  // ElementD
  {{DataTypeTag[t.accumulator]}}, // Accumulator
  {{DataTypeTag[t.b_group_scale]}}, // GroupScaleT
  {{DataTypeTag[t.b_group_zeropoint]}}, // GroupZeroT
  {{DataTypeTag[t.b_channel_scale]}}, // ChannelScaleT
  {{DataTypeTag[t.a_token_scale]}}, // TokenScaleT
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  Sch>;

{% for sch in schs %}
{% set sch_sig = gen_sch_sig(sch) -%}
void 
impl_{{type_sig}}_sch_{{sch_sig}}(MMArgs args) {
  run_impl<Kernel_{{type_sig}}<sch_{{sch_sig}}>>(args);
}
{%- endfor %}
{%- endfor %}

#endif

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels
"""

PREPACK_TEMPLATE = """
/*
 * Copyright 2025 vLLM Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/generate.py
 */

#include "csrc/kernels/nvidia/machete/machete_prepack_launcher.cuh"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace machete {

#if defined(ENABLE_MACHETE)

void prepack_B_dispatch(PrepackBArgs args, void* out_ptr, cudaStream_t stream) {
  auto convert_type = args.maybe_group_scales_type.value_or(args.a_type);
  {%- for t in types %}
  {% set b_type = unsigned_type_with_bitwidth(t.b_num_bits) %}
  if (args.a_type == {{TorchTypeTag[t.a]}}
      && args.b_type.size_bits() == {{t.b_num_bits}} 
      && convert_type == {{TorchTypeTag[t.convert]}}) {
    return prepack_impl<
      PrepackedLayoutBTemplate<
        {{DataTypeTag[t.a]}}, // ElementA
        {{DataTypeTag[b_type]}}, // ElementB
        {{DataTypeTag[t.convert]}}, // ElementConvert
        {{DataTypeTag[t.accumulator]}}, // Accumulator
        cutlass::layout::ColumnMajor,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>
    >(args.B_ptr, args.B_shape, out_ptr, stream);
  }
  {%- endfor %}
  
  KLLM_KERNEL_CHECK_WITH_INFO(false, fmtstr(
    "prepack_B_dispatch(..) is not implemented for atype = {}, b_type = {}, with_group_scales_type= {}",
        args.a_type.str(), args.b_type.str(), args.maybe_group_scales_type ? (*args.maybe_group_scales_type).str() : "None"));
}

#endif

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels
"""

TmaMI = MixedInputKernelScheduleType.TmaWarpSpecializedCooperative
TmaCoop = EpilogueScheduleType.TmaWarpSpecializedCooperative


@dataclass(frozen=True)
class ScheduleConfig:
    tile_shape_mn: tuple[int, int]
    cluster_shape_mnk: tuple[int, int, int]
    kernel_schedule: MixedInputKernelScheduleType
    epilogue_schedule: EpilogueScheduleType
    tile_scheduler: TileSchedulerType


@dataclass(frozen=True)
class TypeConfig:
    a: DataType
    b: Union[DataType, KLLMDataType]
    b_group_scale: DataType
    b_group_zeropoint: DataType
    b_channel_scale: DataType
    a_token_scale: DataType
    out: DataType
    accumulator: DataType


@dataclass(frozen=True)
class PrepackTypeConfig:
    a: DataType
    b_num_bits: int
    convert: DataType
    accumulator: DataType


@dataclass
class ImplConfig:
    types: TypeConfig
    schedules: list[ScheduleConfig]
    heuristic: list[tuple[Optional[str], ScheduleConfig]]


def generate_sch_sig(schedule_config: ScheduleConfig) -> str:
    tile_shape = (
        f"{schedule_config.tile_shape_mn[0]}x{schedule_config.tile_shape_mn[1]}"
    )
    cluster_shape = (f"{schedule_config.cluster_shape_mnk[0]}" +
                     f"x{schedule_config.cluster_shape_mnk[1]}" +
                     f"x{schedule_config.cluster_shape_mnk[2]}")
    kernel_schedule = KLLMKernelScheduleTag[schedule_config.kernel_schedule]\
        .split("::")[-1]
    epilogue_schedule = EpilogueScheduleTag[
        schedule_config.epilogue_schedule].split("::")[-1]
    tile_scheduler = TileSchedulerTag[schedule_config.tile_scheduler]\
        .split("::")[-1]

    return (f"{tile_shape}_{cluster_shape}_{kernel_schedule}" +
            f"_{epilogue_schedule}_{tile_scheduler}")


# mostly unique shorter sch_sig
def generate_terse_sch_sig(schedule_config: ScheduleConfig) -> str:
    kernel_terse_names_replace = {
        "KernelTmaWarpSpecializedCooperative": "TmaMI_",
        "TmaWarpSpecializedCooperative_": "TmaCoop_",
        "StreamKScheduler": "streamK",
    }

    sch_sig = generate_sch_sig(schedule_config)
    for orig, terse in kernel_terse_names_replace.items():
        sch_sig = sch_sig.replace(orig, terse)
    return sch_sig


# unique type_name
def generate_type_signature(kernel_types: TypeConfig):
    return str("".join([
        KLLMDataTypeNames[getattr(kernel_types, field.name)]
        for field in fields(TypeConfig)
    ]))


def generate_type_option_name(kernel_types: TypeConfig):
    return ", ".join([
        f"{field.name.replace('b_', 'with_')+'_type'}=" +
        KLLMDataTypeNames[getattr(kernel_types, field.name)]
        for field in fields(TypeConfig)
    ])


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def to_cute_constant(value: list[int]):

    def _to_cute_constant(value: int):
        if is_power_of_two(value):
            return f"_{value}"
        else:
            return f"Int<{value}>"

    if isinstance(value, Iterable):
        return [_to_cute_constant(value) for value in value]
    else:
        return _to_cute_constant(value)


def unique_schedules(impl_configs: list[ImplConfig]):
    return list(
        set(sch for impl_config in impl_configs
            for sch in impl_config.schedules))


def unsigned_type_with_bitwidth(num_bits):
    return {
        4: DataType.u4,
        8: DataType.u8,
        16: DataType.u16,
        32: DataType.u32,
        64: DataType.u64,
    }[num_bits]


template_globals = {
    "void": DataType.void,
    "DataTypeTag": KLLMDataTypeTag,
    "KLLMScalarTypeTag": KLLMDataTypeKLLMScalarTypeTag,
    "TorchTypeTag": KLLMDataTypeTorchDataTypeTag,
    "KernelScheduleTag": KLLMKernelScheduleTag,
    "EpilogueScheduleTag": EpilogueScheduleTag,
    "TileSchedulerTag": TileSchedulerTag,
    "to_cute_constant": to_cute_constant,
    "gen_sch_sig": generate_terse_sch_sig,
    "gen_type_sig": generate_type_signature,
    "unique_schedules": unique_schedules,
    "unsigned_type_with_bitwidth": unsigned_type_with_bitwidth,
    "gen_type_option_name": generate_type_option_name
}


def create_template(template_str):
    template = jinja2.Template(template_str)
    template.globals.update(template_globals)
    return template


mm_dispatch_template = create_template(DISPATCH_TEMPLATE)
mm_impl_template = create_template(IMPL_TEMPLATE)
prepack_dispatch_template = create_template(PREPACK_TEMPLATE)


def create_sources(impl_configs: list[ImplConfig], num_impl_files=8):
    sources = []

    sources.append((
        "machete_mm_dispatch",
        mm_dispatch_template.render(impl_configs=impl_configs),
    ))

    prepack_types = []
    for impl_config in impl_configs:
        convert_type = impl_config.types.a \
             if impl_config.types.b_group_scale == DataType.void \
             else impl_config.types.b_group_scale
        prepack_types.append(
            PrepackTypeConfig(
                a=impl_config.types.a,
                b_num_bits=KLLMDataTypeSize[impl_config.types.b],
                convert=convert_type,
                accumulator=impl_config.types.accumulator,
            ))

    def prepacked_type_key(prepack_type: PrepackTypeConfig):
        # For now we we can just use the first accumulator type seen since
        # the tensor core shapes/layouts don't vary based on accumulator
        # type so we can generate less code this way
        return (prepack_type.a, prepack_type.b_num_bits, prepack_type.convert)

    unique_prepack_types = []
    prepack_types_seen = set()
    for prepack_type in prepack_types:
        key = prepacked_type_key(prepack_type)
        if key not in prepack_types_seen:
            unique_prepack_types.append(prepack_type)
            prepack_types_seen.add(key)

    sources.append((
        "machete_prepack",
        prepack_dispatch_template.render(types=unique_prepack_types, ),
    ))

    # Split up impls across files
    num_impls = reduce(lambda x, y: x + len(y.schedules), impl_configs, 0)
    num_impls_per_file = math.ceil(num_impls / num_impl_files)

    files_impls: list[list[ImplConfig]] = [[]]

    curr_num_impls_assigned = 0
    curr_impl_in_file = 0
    curr_impl_configs = deepcopy(list(reversed(impl_configs)))

    while curr_num_impls_assigned < num_impls:
        room_left_in_file = num_impls_per_file - curr_impl_in_file
        if room_left_in_file == 0:
            files_impls.append([])
            room_left_in_file = num_impls_per_file
            curr_impl_in_file = 0

        curr_ic = curr_impl_configs[-1]
        if len(curr_ic.schedules) >= room_left_in_file:
            # Break apart the current impl config
            tmp_ic = deepcopy(curr_ic)
            tmp_ic.schedules = curr_ic.schedules[:room_left_in_file]
            curr_ic.schedules = curr_ic.schedules[room_left_in_file:]
            files_impls[-1].append(tmp_ic)
        else:
            files_impls[-1].append(curr_ic)
            curr_impl_configs.pop()
        curr_num_impls_assigned += len(files_impls[-1][-1].schedules)
        curr_impl_in_file += len(files_impls[-1][-1].schedules)

    for part, file_impls in enumerate(files_impls):
        sources.append((
            f"machete_mm_impl_part{part+1}",
            mm_impl_template.render(impl_configs=file_impls),
        ))

    return sources


def generate(output_dir):
    sch_common_params = dict(
        kernel_schedule=TmaMI,
        epilogue_schedule=TmaCoop,
        tile_scheduler=TileSchedulerType.StreamK,
    )

    # Stored as "condition": ((tile_shape_mn), (cluster_shape_mnk))
    default_tile_heuristic_config = {
        #### M = 257+
        "M > 256 && K <= 16384 && N <= 4096": ((128, 128), (2, 1, 1)),
        "M > 256": ((128, 256), (2, 1, 1)),
        #### M = 129-256
        "M > 128 && K <= 4096 && N <= 4096": ((128, 64), (2, 1, 1)),
        "M > 128 && K <= 8192 && N <= 8192": ((128, 128), (2, 1, 1)),
        "M > 128": ((128, 256), (2, 1, 1)),
        #### M = 65-128
        "M > 64 && K <= 4069 && N <= 4069": ((128, 32), (2, 1, 1)),
        "M > 64 && K <= 4069 && N <= 8192": ((128, 64), (2, 1, 1)),
        "M > 64 && K >= 8192 && N >= 12288": ((256, 128), (2, 1, 1)),
        "M > 64": ((128, 128), (2, 1, 1)),
        #### M = 33-64
        "M > 32 && K <= 6144 && N <= 6144": ((128, 16), (1, 1, 1)),
        "M > 32 && K >= 16384 && N >= 12288": ((256, 64), (2, 1, 1)),
        "M > 32": ((128, 64), (2, 1, 1)),
        #### M = 17-32
        "M > 16 && K <= 12288 && N <= 8192": ((128, 32), (2, 1, 1)),
        "M > 16": ((256, 32), (2, 1, 1)),
        #### M = 1-16
        "N >= 26624": ((256, 16), (1, 1, 1)),
        None: ((128, 16), (1, 1, 1)),
    }

    # For now we use the same heuristic for all types
    # Heuristic is currently tuned for H100s
    default_heuristic = [
        (cond, ScheduleConfig(*tile_config,
                              **sch_common_params))  # type: ignore
        for cond, tile_config in default_tile_heuristic_config.items()
    ]

    def get_unique_schedules(heuristic: dict[str, ScheduleConfig]):
        # Do not use schedules = list(set(...)) because we need to make sure
        # the output list is deterministic; otherwise the generated kernel file
        # will be non-deterministic and causes ccache miss.
        schedules = []
        for _, schedule_config in heuristic:
            if schedule_config not in schedules:
                schedules.append(schedule_config)
        return schedules

    impl_configs = []

    GPTQ_kernel_type_configs = list(
        TypeConfig(
            a=a,
            b=b,
            b_group_scale=a,
            b_group_zeropoint=DataType.void,
            b_channel_scale=DataType.void,
            a_token_scale=DataType.void,
            out=a,
            accumulator=DataType.f32,
        ) for b in (KLLMDataType.u4b8, KLLMDataType.u8b128)
        for a in (DataType.f16, DataType.bf16))

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(GPTQ_kernel_type_configs,
                     itertools.repeat(get_unique_schedules(default_heuristic)),
                     itertools.repeat(default_heuristic))
    ]

    AWQ_kernel_type_configs = list(
        TypeConfig(
            a=a,
            b=b,
            b_group_scale=a,
            b_group_zeropoint=a,
            b_channel_scale=DataType.void,
            a_token_scale=DataType.void,
            out=a,
            accumulator=DataType.f32,
        ) for b in (DataType.u4, DataType.u8)
        for a in (DataType.f16, DataType.bf16))

    impl_configs += [
        ImplConfig(x[0], x[1], x[2])
        for x in zip(AWQ_kernel_type_configs,
                     itertools.repeat(get_unique_schedules(default_heuristic)),
                     itertools.repeat(default_heuristic))
    ]

    # TODO: 忽略掉暂时不需要的QQQ量化
    # # Stored as "condition": ((tile_shape_mn), (cluster_shape_mnk))
    # # TODO (LucasWilkinson): Further tuning required
    # qqq_tile_heuristic_config = {
    #     #### M = 257+
    #     # ((128, 256), (2, 1, 1)) Broken for QQQ types
    #     # TODO (LucasWilkinson): Investigate further
    #     # "M > 256 && K <= 16384 && N <= 4096": ((128, 128), (2, 1, 1)),
    #     # "M > 256": ((128, 256), (2, 1, 1)),
    #     "M > 256": ((128, 128), (2, 1, 1)),
    #     #### M = 129-256
    #     "M > 128 && K <= 4096 && N <= 4096": ((128, 64), (2, 1, 1)),
    #     "M > 128 && K <= 8192 && N <= 8192": ((128, 128), (2, 1, 1)),
    #     # ((128, 256), (2, 1, 1)) Broken for QQQ types
    #     # TODO (LucasWilkinson): Investigate further
    #     # "M > 128": ((128, 256), (2, 1, 1)),
    #     "M > 128": ((128, 128), (2, 1, 1)),
    #     #### M = 65-128
    #     "M > 64 && K <= 4069 && N <= 4069": ((128, 32), (2, 1, 1)),
    #     "M > 64 && K <= 4069 && N <= 8192": ((128, 64), (2, 1, 1)),
    #     "M > 64 && K >= 8192 && N >= 12288": ((256, 128), (2, 1, 1)),
    #     "M > 64": ((128, 128), (2, 1, 1)),
    #     #### M = 33-64
    #     "M > 32 && K <= 6144 && N <= 6144": ((128, 16), (1, 1, 1)),
    #     # Broken for QQQ types
    #     # TODO (LucasWilkinson): Investigate further
    #     #"M > 32 && K >= 16384 && N >= 12288": ((256, 64), (2, 1, 1)),
    #     "M > 32": ((128, 64), (2, 1, 1)),
    #     #### M = 17-32
    #     "M > 16 && K <= 12288 && N <= 8192": ((128, 32), (2, 1, 1)),
    #     "M > 16": ((256, 32), (2, 1, 1)),
    #     #### M = 1-16
    #     "N >= 26624": ((256, 16), (1, 1, 1)),
    #     None: ((128, 16), (1, 1, 1)),
    # }

    # # For now we use the same heuristic for all types
    # # Heuristic is currently tuned for H100s
    # qqq_heuristic = [
    #     (cond, ScheduleConfig(*tile_config,
    #                           **sch_common_params))  # type: ignore
    #     for cond, tile_config in qqq_tile_heuristic_config.items()
    # ]

    # QQQ_kernel_types = [
    #     *(TypeConfig(
    #         a=DataType.s8,
    #         b=KLLMDataType.u4b8,
    #         b_group_scale=b_group_scale,
    #         b_group_zeropoint=DataType.void,
    #         b_channel_scale=DataType.f32,
    #         a_token_scale=DataType.f32,
    #         out=DataType.f16,
    #         accumulator=DataType.s32,
    #     ) for b_group_scale in (DataType.f16, DataType.void)),
    #     *(TypeConfig(
    #         a=DataType.e4m3,
    #         b=KLLMDataType.u4b8,
    #         b_group_scale=b_group_scale,
    #         b_group_zeropoint=DataType.void,
    #         b_channel_scale=DataType.f32,
    #         a_token_scale=DataType.f32,
    #         out=DataType.f16,
    #         accumulator=DataType.f32,
    #     ) for b_group_scale in (DataType.f16, DataType.void)),
    # ]

    # impl_configs += [
    #     ImplConfig(x[0], x[1], x[2])
    #     for x in zip(QQQ_kernel_types,
    #                  itertools.repeat(get_unique_schedules(qqq_heuristic)),
    #                  itertools.repeat(qqq_heuristic))
    # ]

    # Delete the "generated" directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create the "generated" directory
    os.makedirs(output_dir)

    # Render each group of configurations into separate files
    for filename, code in create_sources(impl_configs):
        filepath = os.path.join(output_dir, f"{filename}.cu")
        with open(filepath, "w") as output_file:
            output_file.write(code)
        print(f"Rendered template to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output_dir', type=str, required=True, help='The directory where generated files will be saved.')
    args = parser.parse_args()

    generate(args.output_dir)

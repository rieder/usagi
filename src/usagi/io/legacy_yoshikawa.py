# Read Yoshikawa-type economy snapshot (e.g. used in GreeM, for
# CosmoGrid) and save as an Amuse-type hdf5 file.

import os,sys

import argparse

from struct import *
import numpy as np

from amuse.io import read_set_from_file as amuse_read_set_from_file

from amuse.units.core import unit_with_specific_dtype
from amuse.lab import *

UIMAX = (4294967296)   # 2**32
UIMAXINV = 1.0 / UIMAX # 1/2**32


## From http://davidejones.com/blog/1413-python-precision-floating-point/
class Float16Compressor:
    def __init__(self):
        self.temp = 0
    	
    def compress(self,float32):
        F16_EXPONENT_BITS = 0x1F
        F16_EXPONENT_SHIFT = 10
        F16_EXPONENT_BIAS = 15
        F16_MANTISSA_BITS = 0x3ff
        F16_MANTISSA_SHIFT =  (23 - F16_EXPONENT_SHIFT)
        F16_MAX_EXPONENT =  (F16_EXPONENT_BITS << F16_EXPONENT_SHIFT)
        
        a = struct.pack('>f',float32)
        b = binascii.hexlify(a)
        
        f32 = int(b,16)
        f16 = 0
        sign = (f32 >> 16) & 0x8000
        exponent = ((f32 >> 23) & 0xff) - 127
        mantissa = f32 & 0x007fffff
        		
        if exponent == 128:
            f16 = sign | F16_MAX_EXPONENT
            if mantissa:
                f16 |= (mantissa & F16_MANTISSA_BITS)
        elif exponent > 15:
            f16 = sign | F16_MAX_EXPONENT
        elif exponent > -15:
            exponent += F16_EXPONENT_BIAS
            mantissa >>= F16_MANTISSA_SHIFT
            f16 = sign | exponent << F16_EXPONENT_SHIFT | mantissa
        else:
            f16 = sign
        return f16
    	
    def decompress(self,float16):
        s = int((float16 >> 15) & 0x00000001)    # sign
        e = int((float16 >> 10) & 0x0000001f)    # exponent
        f = int(float16 & 0x000003ff)            # fraction
        
        if e == 0:
            if f == 0:
                return int(s << 31)
            else:
                while not (f & 0x00000400):
                    f = f << 1
                    e -= 1
                e += 1
                f &= ~0x00000400
                #print(s,e,f)
        elif e == 31:
            if f == 0:
                return int((s << 31) | 0x7f800000)
            else:
                return int((s << 31) | 0x7f800000 | (f << 13))
        
        e = e + (127 -15)
        f = f << 13
        return int((s << 31) | (e << 23) | f)

def f16tof32(f):
    fcomp = Float16Compressor()
    return unpack('f', pack('I', fcomp.decompress(f)))[0]

def p160Toyp(
        raw,
        ):
    p160    = Struct('I I I I I')

    tmp = p160.unpack(raw) 
    
    ix = np.uint32(tmp[0] & 0xFFFFFF00)
    iy = np.uint32((tmp[0] & 0x000000FF) << 24 | (tmp[1] & 0xFFFF0000) >> 8)
    iz = np.uint32((tmp[1] & 0x0000FFFF) << 16 | (tmp[2] & 0xFF000000) >> 16)
    
    vx = ((tmp[2] & 0x00FFFF00) >> 8)
    vy = ((tmp[2] & 0x000000FF) << 8 | (tmp[3] & 0xFF000000) >> 24)
    vz = ((tmp[3] & 0x00FFFF00) >> 8)
    
    xpos = np.float64(ix) * UIMAXINV
    ypos = np.float64(iy) * UIMAXINV
    zpos = np.float64(iz) * UIMAXINV
    
    xvel    = f16tof32(vx)
    yvel    = f16tof32(vy)
    zvel    = f16tof32(vz)
    
    u = tmp[3] & 0xff
    l = tmp[4]
    old_index = l + u*2**32

    return old_index, xpos, ypos, zpos, xvel, yvel, zvel

def bigtolittle(raw, bittype):
    tmp = unpack('>%s'%bittype,raw)
    return tmp

def read_cosmogrid_header(name,particles):
    infile  = open(name, 'r')
    particles.collection_attributes.step    = int(infile.readline().split()[1])
    particles.collection_attributes.znow    = float(infile.readline().split()[1])
    particles.collection_attributes.anow    = float(infile.readline().split()[1])
    particles.collection_attributes.tnow    = float(infile.readline().split()[1])
    particles.collection_attributes.omega0  = float(infile.readline().split()[1])
    particles.collection_attributes.omegab  = float(infile.readline().split()[1])
    particles.collection_attributes.lambda0 = float(infile.readline().split()[1])
    particles.collection_attributes.hubble  = float(infile.readline().split()[1])
    particles.collection_attributes.astart  = float(infile.readline().split()[1])
    particles.collection_attributes.lunit   = float(infile.readline().split()[1])
    particles.collection_attributes.munit   = float(infile.readline().split()[1])
    particles.collection_attributes.tunit   = float(infile.readline().split()[1])
    particles.collection_attributes.mass    = float(infile.readline().split()[1])
    
    particles.collection_attributes.a       = particles.collection_attributes.anow
    particles.collection_attributes.redshift= particles.collection_attributes.znow
    particles.collection_attributes.h       = particles.collection_attributes.hubble
    particles.collection_attributes.boxsize = particles.collection_attributes.lunit
    #particles.collection_attributes.ntotal    = ntotal
    particles.collection_attributes.omega   = (
            particles.collection_attributes.omega0 + 
            particles.collection_attributes.lambda0
            )
    particles.collection_attributes.omegal  = particles.collection_attributes.lambda0
    return

def read_set_from_file(
        filename,
        format = "yoshikawa",
        header = False,
        **format_specific_keyword_arguments
        ):
    if format != "yoshikawa":
        # pass on to Amuse
        p   = amuse_read_set_from_file(
                filename,
                format = format,
                **format_specific_keyword_arguments)
        return p
    if not header:
        tmp = filename.split("-")
        if len(tmp) == 1:
            tmp = filename.split(".")
        if len(tmp) == 1:
            print "header is the same as file, that doesn't work"
            exit()
        header  = ''.join(tmp[0:-1])

    infile = open(filename, 'rb')

    dataheader      = unpack('i',infile.read(4))[0]

    # Now, we must interpret the header.
    # If we assume the file is an economy snapshot,
    # 10 means little endian and other means big endian.
    if dataheader == 10:
        bigendian   = False
    else:
        bigendian   = True

    raw = infile.read(4)
    if bigendian:
        npart   = bigtolittle(raw, "i")[0]
    else:
        npart   = unpack("i",raw)[0]

    keys    = []
    xpos    = []
    ypos    = []
    zpos    = []
    xvel    = []
    yvel    = []
    zvel    = []

    for i in range(npart):
        # Read bits the size of 1 particle (160 bits / 20 bytes)
        tmp = infile.read(20)
        # swap endianness?

        p = p160Toyp(tmp)
        keys.append(p[0])
        xpos.append(p[1])
        ypos.append(p[2])
        zpos.append(p[3])
        xvel.append(p[4])
        yvel.append(p[5])
        zvel.append(p[6])

    block   = Particles(keys=keys)

    read_cosmogrid_header(header,block)

    xpos    = np.array(xpos) * (
            block.collection_attributes.lunit 
            ) | unit_with_specific_dtype(units.Mpc, np.float32)
    ypos    = np.array(ypos) * (
            block.collection_attributes.lunit 
            ) | unit_with_specific_dtype(units.Mpc, np.float32)
    zpos    = np.array(zpos) * (
            block.collection_attributes.lunit 
            ) | unit_with_specific_dtype(units.Mpc, np.float32)
    vunit   = (
            block.collection_attributes.anow /
            block.collection_attributes.hubble *
            (block.collection_attributes.lunit | units.Mpc) / 
            (block.collection_attributes.tunit | units.s)
            ).value_in(units.kms)
    xvel    = np.array(xvel) * (
            vunit 
            ) | unit_with_specific_dtype(units.kms, np.float16)
    yvel    = np.array(yvel) * (
            vunit 
            ) | unit_with_specific_dtype(units.kms, np.float16)
    zvel    = np.array(zvel) * (
            vunit 
            ) | unit_with_specific_dtype(units.kms, np.float16)
    mass    = (
            block.collection_attributes.munit * 
            block.collection_attributes.mass 
            ) | unit_with_specific_dtype(units.MSun, np.float32)


    block.x     = xpos 
    block.y     = ypos 
    block.z     = zpos 
    block.vx    = xvel
    block.vy    = yvel
    block.vz    = zvel

    # Writing mass only to the collection_attributes would save 4
    # bytes per particle...
    # block.collection_attributes.particle_mass = mass
    block.mass  = mass

    return block

def new_argument_parser():
    parser  = argparse.ArgumentParser()

    parser.add_argument(
            '-i',
            dest    = 'snapshotname',
            type    = str,
            default = "snapshot.eco",
            help    = "Economy-type Yoshikawa snapshot to read (default: %s)"%(
                "snapshot.eco",
                ),
            )
    parser.add_argument(
            '-H',
            dest    = 'headername',
            type    = str,
            default = "",
            help    = "Yoshikawa snapshot header to read (default: snapshotname minus part after '.' or '-')",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args    = new_argument_parser()
    snapshotname    = args.snapshotname

    if args.headername != "":
        headername      = args.headername
    else:
        tmp = snapshotname.split("-")
        if len(tmp) == 1:
            tmp = snapshotname.split(".")
        if len(tmp) == 1:
            print "header is the same as file, that doesn't work"
            exit()
        headername  = ''.join(tmp[0:-1])

    block   = read_cosmogrid_snapshot(snapshotname, headername)

    write_set_to_file(
            block,
            snapshotname + ".hdf5",
            'amuse',
            )


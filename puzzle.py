#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
puzzle.py - Recherche de clés privées Bitcoin (secp256k1) dans une plage connue : multi-GPU (CUDA) + multi-cœurs.

Ce fichier est AUTONOME : il contient tout le code source (C++ / CUDA), le compile au premier lancement
(GPU si nvcc est présent, sinon CPU seul) puis lance la recherche. Les arguments sont ceux du programme :

    python3 puzzle.py <clé_publique | adresse | cibles.txt> <bits | début:fin> [address] [mode] [options]

Exemples :
    python3 puzzle.py --gpuinfo
    python3 puzzle.py $(python3 puzzle.py --genpub C3A5B7D1) 32
    python3 puzzle.py 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH 400000000:7ffffffff address -s

Commandes du lanceur : --build (compiler seulement)  --rebuild (forcer)  --build-cpu (version CPU seule)
                       --extract DOSSIER (écrire les sources)  --help
Voir le README pour toutes les options (-nogpu, -gpus, -t, -ckpt, -fresh, -out, -log ...).
"""
import hashlib
import os
import shutil
import subprocess
import sys

SOURCES = {}
SOURCES['puzzle.cpp'] = r'''// puzzle.cpp - Recherche de clés privées Bitcoin (secp256k1) multi-cibles, multi-cœurs ET multi-GPU (CUDA).
//
// Compilation : python3 puzzle.py --build   (détecte nvcc / les cartes ; sans CUDA : version CPU seule)
//   à la main, CPU seul : g++ -O3 -march=native -std=c++17 -pthread -DNO_GPU puzzle.cpp gpu_stub.cpp -o puzzle
//   Windows/MinGW (CPU) : si le programme s'arrête brutalement, ajoute -mno-avx à la compilation.
//
// Usage :
//   puzzle <clé_publique | adresse | cibles.txt | cibles.bloom> <bits | début:fin> [address] [mode] [options]
//   puzzle --genpub <clé_privée_hex> | --genaddr <clé_privée_hex> | --gpuinfo       (utilitaires)
//
// La clé recherchée est dans [2^(bits-1) , 2^bits - 1], ou dans la plage début:fin donnée en hexadécimal
// (n'importe quelle plage, ex. 2000000:3ffffff).
// address      : à ajouter quand les cibles sont des adresses Bitcoin (1... ou bc1q...)
// cibles.txt   : une cible par ligne -> construit et sauvegarde cibles.bloom ; cibles.bloom : chargement direct
// Mode (un seul par lancement ; tous les calculateurs travaillent ensemble sur la même suite de blocs) :
//   -r aléatoire (défaut)   -s séquentiel   -sobol   -pi   -e   -ln   -m Monte Carlo
// Calculateurs :
//   (défaut)    toutes les cartes graphiques NVIDIA détectées ; sans carte : tous les cœurs CPU
//   -nogpu      CPU uniquement            -gpus 0,1   n'utiliser que ces cartes
//   -t N        N cœurs CPU en plus des GPU (avec GPU : 0 par défaut ; sans GPU : tous)
// Reprise (sessions Colab / Kaggle qui s'arrêtent) :
//   la progression est sauvegardée toutes les 60 s (puzzle_<empreinte>.ckpt) et reprise automatiquement
//   au relancement de la même commande. -fresh : repartir de zéro   -nockpt : désactiver
//   -ckpt FICHIER : fichier de reprise   -every S : période de sauvegarde (secondes)
// Divers : -out FICHIER (résultats, ex. sur Google Drive)   -log S (période d'affichage hors terminal, défaut 10 s)
//          -gjb N (lots par travail GPU, défaut 16)
#define _FILE_OFFSET_BITS 64
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <atomic>
#include <exception>
#include <ctime>
#include <functional>
#include <deque>
#include "gpu_api.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <thread>
#include <unistd.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;

struct U256 { u64 d[4]; };               // little-endian (d[0] = poids faible)
static const U256 ZERO = {{0,0,0,0}};
static const U256 ONE  = {{1,0,0,0}};
static const U256 PRIME = {{0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL}};
static const u64  KC = 0x1000003D1ULL;   // 2^256 mod p

// ---------------------------------------------------------------- 256 bits brut
static inline bool is_zero(const U256& a){ return (a.d[0]|a.d[1]|a.d[2]|a.d[3])==0; }
static inline bool eq(const U256& a,const U256& b){
    return ((a.d[0]^b.d[0])|(a.d[1]^b.d[1])|(a.d[2]^b.d[2])|(a.d[3]^b.d[3]))==0;
}
static inline bool geq(const U256& a,const U256& b){
    for(int i=3;i>=0;--i){ if(a.d[i]!=b.d[i]) return a.d[i]>b.d[i]; }
    return true;
}
static inline u64 add_raw(U256& r,const U256& a,const U256& b){
    u128 acc=0; u64 c=0;
    for(int i=0;i<4;i++){ acc=(u128)a.d[i]+b.d[i]+c; r.d[i]=(u64)acc; c=(u64)(acc>>64); }
    return c;
}
static inline u64 sub_raw(U256& r,const U256& a,const U256& b){
    u64 br=0;
    for(int i=0;i<4;i++){
        u128 t=(u128)a.d[i]-b.d[i]-br;
        r.d[i]=(u64)t; br=(u64)((t>>64)&1);
    }
    return br;
}
static inline void add_small(U256& a,u64 v){
    u128 acc=(u128)a.d[0]+v; a.d[0]=(u64)acc; u64 c=(u64)(acc>>64);
    for(int i=1;i<4;i++){ acc=(u128)a.d[i]+c; a.d[i]=(u64)acc; c=(u64)(acc>>64); }
}
static inline int get_bit(const U256& a,int i){ return (int)((a.d[i>>6]>>(i&63))&1); }
static int highest_bit(const U256& a){
    for(int i=255;i>=0;--i) if(get_bit(a,i)) return i;
    return -1;
}
static U256 shr2(U256 a){
    for(int i=0;i<4;i++){
        a.d[i]>>=2;
        if(i<3) a.d[i]|=a.d[i+1]<<62;   // a.d[i+1] n'a pas encore été décalé
    }
    return a;
}

// ---------------------------------------------------------------- hex / décimal
static bool parse_hex(const std::string& s,U256& out){
    if(s.empty()||s.size()>64) return false;
    out=ZERO;
    for(size_t i=0;i<s.size();i++){
        char c=s[s.size()-1-i]; int v;
        if(c>='0'&&c<='9') v=c-'0'; else if(c>='a'&&c<='f') v=c-'a'+10;
        else if(c>='A'&&c<='F') v=c-'A'+10; else return false;
        out.d[i/16]|=(u64)v<<((i%16)*4);
    }
    return true;
}
static std::string to_hex(const U256& a){
    char buf[65];
    snprintf(buf,sizeof buf,"%016llx%016llx%016llx%016llx",
        (unsigned long long)a.d[3],(unsigned long long)a.d[2],
        (unsigned long long)a.d[1],(unsigned long long)a.d[0]);
    return buf;
}
static std::string to_dec(U256 a){
    if(is_zero(a)) return "0";
    std::vector<u64> parts;
    while(!is_zero(a)){
        u128 rem=0;
        for(int i=3;i>=0;--i){
            u128 cur=(rem<<64)|a.d[i];
            a.d[i]=(u64)(cur/1000000000ULL);
            rem=cur%1000000000ULL;
        }
        parts.push_back((u64)rem);
    }
    char buf[32]; std::string s;
    snprintf(buf,sizeof buf,"%llu",(unsigned long long)parts.back()); s+=buf;
    for(int i=(int)parts.size()-2;i>=0;--i){ snprintf(buf,sizeof buf,"%09llu",(unsigned long long)parts[i]); s+=buf; }
    return s;
}

// ---------------------------------------------------------------- corps Fp (p = 2^256 - 2^32 - 977)
static inline void fe_add(U256& r,const U256& a,const U256& b){
    U256 t; u64 c=add_raw(t,a,b);
    if(c){ U256 k={{KC,0,0,0}}; add_raw(t,t,k); }
    if(geq(t,PRIME)) sub_raw(t,t,PRIME);
    r=t;
}
static inline void fe_sub(U256& r,const U256& a,const U256& b){
    U256 t; u64 br=sub_raw(t,a,b);
    if(br){ U256 k={{KC,0,0,0}}; sub_raw(t,t,k); }
    r=t;
}
// Réduction d'un produit de 512 bits modulo p (2^256 = 2^32 + 977 mod p)
static inline void fe_reduce(U256& r,const u64* t){
    u64 r5[5]; u64 carry=0;
    for(int i=0;i<4;i++){
        u128 acc=(u128)t[4+i]*KC+t[i]+carry;
        r5[i]=(u64)acc; carry=(u64)(acc>>64);
    }
    r5[4]=carry;
    U256 res; u128 acc=(u128)r5[4]*KC+r5[0];
    res.d[0]=(u64)acc; carry=(u64)(acc>>64);
    for(int i=1;i<4;i++){ acc=(u128)r5[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    if(carry){
        acc=(u128)res.d[0]+KC; res.d[0]=(u64)acc; carry=(u64)(acc>>64);
        for(int i=1;i<4;i++){ acc=(u128)res.d[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    }
    if(geq(res,PRIME)) sub_raw(res,res,PRIME);
    r=res;
}
static inline void fe_mul(U256& r,const U256& a,const U256& b){
    u64 t[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){
        u64 carry=0;
        for(int j=0;j<4;j++){
            u128 acc=(u128)a.d[i]*b.d[j]+t[i+j]+carry;
            t[i+j]=(u64)acc; carry=(u64)(acc>>64);
        }
        t[i+4]=carry;
    }
    fe_reduce(r,t);
}
// Carré dédié : 10 produits de 64 bits au lieu de 16
static inline void fe_sqr(U256& r,const U256& a){
    u64 t[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<3;i++){
        u64 carry=0;
        for(int j=i+1;j<4;j++){
            u128 acc=(u128)a.d[i]*a.d[j]+t[i+j]+carry;
            t[i+j]=(u64)acc; carry=(u64)(acc>>64);
        }
        t[i+4]=carry;
    }
    u64 c=0;
    for(int k=0;k<8;k++){ u64 v=t[k]; t[k]=(v<<1)|c; c=v>>63; }
    u64 carry=0;
    for(int i=0;i<4;i++){
        u128 acc=(u128)a.d[i]*a.d[i]+t[2*i]+carry;
        t[2*i]=(u64)acc; carry=(u64)(acc>>64);
        acc=(u128)t[2*i+1]+carry;
        t[2*i+1]=(u64)acc; carry=(u64)(acc>>64);
    }
    fe_reduce(r,t);
}
static inline void fe_sqr_n(U256& r,const U256& a,int n){
    U256 t=a; for(int i=0;i<n;i++) fe_sqr(t,t); r=t;
}
static U256 fe_pow(const U256& base,const U256& e){
    U256 r=ONE; int top=highest_bit(e);
    for(int i=top;i>=0;--i){ fe_sqr(r,r); if(get_bit(e,i)) fe_mul(r,r,base); }
    return r;
}
// Inverse modulaire a^(p-2) par chaîne d'additions : 255 carrés + 15 produits
static U256 fe_inv(const U256& a){
    U256 x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    fe_sqr(x2,a); fe_mul(x2,x2,a);
    fe_sqr(x3,x2); fe_mul(x3,x3,a);
    fe_sqr_n(x6,x3,3); fe_mul(x6,x6,x3);
    fe_sqr_n(x9,x6,3); fe_mul(x9,x9,x3);
    fe_sqr_n(x11,x9,2); fe_mul(x11,x11,x2);
    fe_sqr_n(x22,x11,11); fe_mul(x22,x22,x11);
    fe_sqr_n(x44,x22,22); fe_mul(x44,x44,x22);
    fe_sqr_n(x88,x44,44); fe_mul(x88,x88,x44);
    fe_sqr_n(x176,x88,88); fe_mul(x176,x176,x88);
    fe_sqr_n(x220,x176,44); fe_mul(x220,x220,x44);
    fe_sqr_n(x223,x220,3); fe_mul(x223,x223,x3);
    fe_sqr_n(t,x223,23); fe_mul(t,t,x22);
    fe_sqr_n(t,t,5); fe_mul(t,t,a);
    fe_sqr_n(t,t,3); fe_mul(t,t,x2);
    fe_sqr_n(t,t,2); fe_mul(t,t,a);
    return t;
}

// ---------------------------------------------------------------- courbe y^2 = x^3 + 7
struct Aff { U256 x,y; bool inf; };
struct Jac { U256 X,Y,Z; bool inf; };

static Aff GEN;
static const char* GX_HEX="79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
static const char* GY_HEX="483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

static bool on_curve(const Aff& P){
    U256 l,r; fe_sqr(l,P.y); fe_sqr(r,P.x); fe_mul(r,r,P.x);
    U256 seven={{7,0,0,0}}; fe_add(r,r,seven);
    return eq(l,r);
}

// --- Arithmétique affine (lente, 1 inversion par opération) : sert à la vérification indépendante
static Aff aff_add(const Aff& P,const Aff& Q){
    if(P.inf) return Q;
    if(Q.inf) return P;
    U256 l;
    if(eq(P.x,Q.x)){
        if(!eq(P.y,Q.y) || is_zero(P.y)){ Aff o; o.x=ZERO;o.y=ZERO;o.inf=true; return o; }
        U256 n,d; fe_sqr(n,P.x); U256 n3; fe_add(n3,n,n); fe_add(n3,n3,n);   // 3x^2
        fe_add(d,P.y,P.y);
        fe_mul(l,n3,fe_inv(d));
    } else {
        U256 n,d; fe_sub(n,Q.y,P.y); fe_sub(d,Q.x,P.x);
        fe_mul(l,n,fe_inv(d));
    }
    Aff R; R.inf=false;
    fe_sqr(R.x,l); fe_sub(R.x,R.x,P.x); fe_sub(R.x,R.x,Q.x);
    U256 t; fe_sub(t,P.x,R.x); fe_mul(R.y,l,t); fe_sub(R.y,R.y,P.y);
    return R;
}
static Aff aff_mul(const U256& k,const Aff& P){
    Aff R; R.x=ZERO;R.y=ZERO;R.inf=true;
    for(int i=highest_bit(k);i>=0;--i){
        R=aff_add(R,R);
        if(get_bit(k,i)) R=aff_add(R,P);
    }
    return R;
}

// --- Arithmétique Jacobienne (rapide) : sert au calcul du point de départ
static Jac jac_dbl(const Jac& P){
    if(P.inf||is_zero(P.Y)){ Jac o; o.X=o.Y=o.Z=ZERO; o.inf=true; return o; }
    U256 A,B,C,D,E,F,t,t2;
    fe_sqr(A,P.X); fe_sqr(B,P.Y); fe_sqr(C,B);
    fe_add(t,P.X,B); fe_sqr(t,t); fe_sub(t,t,A); fe_sub(t,t,C); fe_add(D,t,t);
    fe_add(E,A,A); fe_add(E,E,A);
    fe_sqr(F,E);
    Jac R; R.inf=false;
    fe_add(t,D,D); fe_sub(R.X,F,t);
    fe_sub(t,D,R.X); fe_mul(t,E,t);
    fe_add(t2,C,C); fe_add(t2,t2,t2); fe_add(t2,t2,t2);   // 8C
    fe_sub(R.Y,t,t2);
    fe_mul(t,P.Y,P.Z); fe_add(R.Z,t,t);
    return R;
}
static Jac jac_madd(const Jac& P,const Aff& Q){
    if(P.inf){ Jac R; R.X=Q.x;R.Y=Q.y;R.Z=ONE;R.inf=false; return R; }
    U256 Z1Z1,U2,S2,H,HH,I,J,r,V,t;
    fe_sqr(Z1Z1,P.Z); fe_mul(U2,Q.x,Z1Z1);
    fe_mul(S2,Q.y,P.Z); fe_mul(S2,S2,Z1Z1);
    fe_sub(H,U2,P.X);
    fe_sub(t,S2,P.Y); fe_add(r,t,t);
    if(is_zero(H)){
        if(is_zero(r)) return jac_dbl(P);
        Jac o; o.X=o.Y=o.Z=ZERO; o.inf=true; return o;
    }
    fe_sqr(HH,H); fe_add(I,HH,HH); fe_add(I,I,I);
    fe_mul(J,H,I); fe_mul(V,P.X,I);
    Jac R; R.inf=false;
    fe_sqr(R.X,r); fe_sub(R.X,R.X,J); fe_sub(R.X,R.X,V); fe_sub(R.X,R.X,V);
    fe_sub(t,V,R.X); fe_mul(t,r,t);
    U256 u; fe_mul(u,P.Y,J); fe_add(u,u,u);
    fe_sub(R.Y,t,u);
    fe_add(t,P.Z,H); fe_sqr(t,t); fe_sub(t,t,Z1Z1); fe_sub(R.Z,t,HH);
    return R;
}
static Jac jac_mul(const U256& k,const Aff& P){
    Jac R; R.X=R.Y=R.Z=ZERO; R.inf=true;
    for(int i=highest_bit(k);i>=0;--i){
        R=jac_dbl(R);
        if(get_bit(k,i)) R=jac_madd(R,P);
    }
    return R;
}
static Aff jac_to_aff(const Jac& J){
    Aff A; A.x=ZERO;A.y=ZERO;
    if(J.inf){ A.inf=true; return A; }
    U256 zi=fe_inv(J.Z),zi2,zi3;
    fe_sqr(zi2,zi); fe_mul(zi3,zi2,zi);
    fe_mul(A.x,J.X,zi2); fe_mul(A.y,J.Y,zi3); A.inf=false;
    return A;
}

// ---------------------------------------------------------------- utilitaires divers
static std::string compress_hex(const Aff& P){
    return std::string(((P.y.d[0]&1)?"03":"02"))+to_hex(P.x);
}
static inline u64 splitmix(u64& s){
    u64 z=(s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
struct Rng {
    u64 s[4];
    void seed(u64 v){ for(int i=0;i<4;i++) s[i]=splitmix(v); }
    static inline u64 rotl(u64 x,int k){ return (x<<k)|(x>>(64-k)); }
    u64 next(){
        u64 r=rotl(s[1]*5,7)*9, t=s[1]<<17;
        s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl(s[3],45);
        return r;
    }
};
struct SpinLock {
    std::atomic_flag f=ATOMIC_FLAG_INIT;
    void lock(){ while(f.test_and_set(std::memory_order_acquire)){} }
    void unlock(){ f.clear(std::memory_order_release); }
};
static U256 shl_u64(u64 v,int sh){
    U256 r=ZERO; int limb=sh>>6, off=sh&63;
    if(limb<4) r.d[limb]|=v<<off;
    if(off && limb+1<4) r.d[limb+1]|=v>>(64-off);
    return r;
}
static U256 low_mask(int nbits){           // 2^nbits - 1
    U256 r=ZERO;
    for(int b=0;b<nbits;b++) r.d[b>>6]|=1ULL<<(b&63);
    return r;
}
static std::string fmt_int(u64 v){
    std::string s=std::to_string(v), o;
    int c=0;
    for(int i=(int)s.size()-1;i>=0;--i){ o.insert(o.begin(),s[i]); if(++c%3==0&&i>0) o.insert(o.begin(),' '); }
    return o;
}
static std::string fmt_keys(long double v){
    const char* u[]={"","K","M","G","T","P","E"};
    int i=0; while(v>=1000.0L && i<6){ v/=1000.0L; i++; }
    char b[32]; snprintf(b,sizeof b,i?"%.2Lf %s":"%.0Lf %s",v,u[i]); return b;
}
static std::string fmt_time(long double s){
    char b[64];
    if(s<0||s!=s) return "?";
    if(s>3.15e9L) return ">100 ans";
    unsigned long long t=(unsigned long long)s;
    unsigned long long y=t/31536000ULL; t%=31536000ULL;
    unsigned long long d=t/86400, h=(t%86400)/3600, m=(t%3600)/60, sec=t%60;
    if(y) snprintf(b,sizeof b,"%llu an(s) %llu j",y,d);
    else if(d) snprintf(b,sizeof b,"%llu j %02lluh %02llum",d,h,m);
    else snprintf(b,sizeof b,"%02llu:%02llu:%02llu",h,m,sec);
    return b;
}
static std::string short_hex(const U256& v){
    std::string h=to_hex(v); size_t p=h.find_first_not_of('0');
    h=(p==std::string::npos)?"0":h.substr(p);
    if(h.size()>22) h=h.substr(0,10)+"…"+h.substr(h.size()-10);
    return "0x"+h;
}

// ---------------------------------------------------------------- SHA-256 / RIPEMD-160 (un seul bloc) et adresses
static inline u32 rotr32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
static inline u32 rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
static const u32 SHA_K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

// SHA-256 d'un message tenant dans un seul bloc (w = 16 mots big-endian déjà préparés)
static inline void sha256_block(const u32* w16,u32* out){
    u32 w[64];
    for(int i=0;i<16;i++) w[i]=w16[i];
    #pragma GCC unroll 64
    for(int i=16;i<64;i++){
        u32 s0=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma GCC unroll 64
    for(int i=0;i<64;i++){
        u32 S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25), ch=(e&f)^(~e&g);
        u32 t1=h+S1+ch+SHA_K[i]+w[i];
        u32 S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22), mj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    out[0]=a+0x6a09e667; out[1]=b+0xbb67ae85; out[2]=c+0x3c6ef372; out[3]=d+0xa54ff53a;
    out[4]=e+0x510e527f; out[5]=f+0x9b05688c; out[6]=g+0x1f83d9ab; out[7]=h+0x5be0cd19;
}
// SHA-256 d'un message de n <= 55 octets (octet par octet)
static void sha256_bytes(const unsigned char* m,size_t n,unsigned char* out32){
    unsigned char buf[64]; memset(buf,0,64); memcpy(buf,m,n); buf[n]=0x80;
    u64 bl=(u64)n*8; for(int i=0;i<8;i++) buf[63-i]=(unsigned char)(bl>>(8*i));
    u32 w[16],o[8];
    for(int i=0;i<16;i++) w[i]=((u32)buf[4*i]<<24)|((u32)buf[4*i+1]<<16)|((u32)buf[4*i+2]<<8)|buf[4*i+3];
    sha256_block(w,o);
    for(int i=0;i<8;i++){ out32[4*i]=o[i]>>24; out32[4*i+1]=o[i]>>16; out32[4*i+2]=o[i]>>8; out32[4*i+3]=o[i]; }
}

static const unsigned char RMD_R1[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
static const unsigned char RMD_R2[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
static const unsigned char RMD_S1[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
static const unsigned char RMD_S2[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
static const u32 RMD_K1[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
static const u32 RMD_K2[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
template<int F> static inline u32 rmd_f(u32 x,u32 y,u32 z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
// RIPEMD-160 d'un message tenant dans un seul bloc (X = 16 mots little-endian)
template<int F> static inline void rmd_seg(const u32* X,u32& a,u32& b,u32& c,u32& d,u32& e,
                                           u32& a2,u32& b2,u32& c2,u32& d2,u32& e2){
    #pragma GCC unroll 16
    for(int j=16*F;j<16*F+16;j++){
        u32 t=rotl32(a+rmd_f<F>(b,c,d)+X[RMD_R1[j]]+RMD_K1[F],RMD_S1[j])+e;
        a=e; e=d; d=rotl32(c,10); c=b; b=t;
        t=rotl32(a2+rmd_f<4-F>(b2,c2,d2)+X[RMD_R2[j]]+RMD_K2[F],RMD_S2[j])+e2;
        a2=e2; e2=d2; d2=rotl32(c2,10); c2=b2; b2=t;
    }
}
static inline void rmd160_block(const u32* X,u32* out){
    u32 a=0x67452301,b=0xEFCDAB89,c=0x98BADCFE,d=0x10325476,e=0xC3D2E1F0;
    u32 a2=a,b2=b,c2=c,d2=d,e2=e;
    rmd_seg<0>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<1>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<2>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<3>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<4>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    const u32 I0=0x67452301u, I1=0xEFCDAB89u, I2=0x98BADCFEu, I3=0x10325476u, I4=0xC3D2E1F0u;
    u32 t=I1+c+d2;
    out[1]=I2+d+e2; out[2]=I3+e+a2; out[3]=I4+a+b2; out[4]=I0+b+c2; out[0]=t;
}
// hash160 empaqueté dans un U256 : lecture little-endian des 20 octets du condensat
static inline U256 pack_rmd(const u32* h){
    U256 r; r.d[0]=h[0]|((u64)h[1]<<32); r.d[1]=h[2]|((u64)h[3]<<32); r.d[2]=h[4]; r.d[3]=0; return r;
}
static U256 pack20(const unsigned char* b){
    U256 r=ZERO; memcpy(&r.d[0],b,8); memcpy(&r.d[1],b+8,8); u32 t; memcpy(&t,b+16,4); r.d[2]=t; return r;
}
static void unpack20(const U256& h,unsigned char* b){
    memcpy(b,&h.d[0],8); memcpy(b+8,&h.d[1],8); u32 t=(u32)h.d[2]; memcpy(b+16,&t,4);
}
// Chemin rapide (boucle de recherche) : hash160 de la clé publique compressée (parité de Y, X)
static inline U256 hash160_fast(const U256& x,u64 par){
    u32 X[8];
    X[0]=(u32)(x.d[3]>>32); X[1]=(u32)x.d[3]; X[2]=(u32)(x.d[2]>>32); X[3]=(u32)x.d[2];
    X[4]=(u32)(x.d[1]>>32); X[5]=(u32)x.d[1]; X[6]=(u32)(x.d[0]>>32); X[7]=(u32)x.d[0];
    u32 w[16],H[8],rw[16],o[5];
    w[0]=((0x02u+(u32)par)<<24)|(X[0]>>8);
    for(int i=1;i<8;i++) w[i]=(X[i-1]<<24)|(X[i]>>8);
    w[8]=(X[7]<<24)|0x00800000u;
    for(int i=9;i<15;i++) w[i]=0;
    w[15]=264;
    sha256_block(w,H);
    for(int i=0;i<8;i++) rw[i]=__builtin_bswap32(H[i]);
    rw[8]=0x80; for(int i=9;i<14;i++) rw[i]=0; rw[14]=256; rw[15]=0;
    rmd160_block(rw,o);
    return pack_rmd(o);
}
// Chemin "octets" (vérifications indépendantes)
static void pub33(const U256& x,u64 par,unsigned char* out){
    out[0]=(unsigned char)(2+par);
    for(int i=0;i<32;i++) out[1+i]=(unsigned char)(x.d[3-i/8]>>(56-8*(i%8)));
}
static void hash160_bytes(const unsigned char* m,size_t n,unsigned char* out20){
    unsigned char dg[32]; sha256_bytes(m,n,dg);
    u32 X[16]={0},o[5];
    for(int i=0;i<8;i++) X[i]=(u32)dg[4*i]|((u32)dg[4*i+1]<<8)|((u32)dg[4*i+2]<<16)|((u32)dg[4*i+3]<<24);
    X[8]=0x80; X[14]=256;
    rmd160_block(X,o);
    for(int i=0;i<5;i++){ out20[4*i]=o[i]; out20[4*i+1]=o[i]>>8; out20[4*i+2]=o[i]>>16; out20[4*i+3]=o[i]>>24; }
}
static U256 hash160_point_bytes(const Aff& P){
    unsigned char pb[33],h[20]; pub33(P.x,P.y.d[0]&1,pb); hash160_bytes(pb,33,h); return pack20(h);
}

// ---- hash160 de plusieurs clés à la fois (vecteurs SIMD via les extensions GCC/Clang)
#if defined(VLANES)
#elif defined(_WIN32)
#define VLANES 4          // 128 bits : évite les soucis d'alignement de pile 32 octets de MinGW
#else
#define VLANES 8          // 256 bits (AVX2 si -march=native)
#endif
typedef u32 vu __attribute__((vector_size(4*VLANES)));
static inline vu vsplat(u32 x){ vu v; for(int i=0;i<VLANES;i++) v[i]=x; return v; }
#define VROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define VROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
template<int F> static inline vu rmd_fv(vu x,vu y,vu z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
template<int F> static inline void rmd_segv(const vu* X,vu& a,vu& b,vu& c,vu& d,vu& e,vu& a2,vu& b2,vu& c2,vu& d2,vu& e2){
    #pragma GCC unroll 16
    for(int j=16*F;j<16*F+16;j++){
        vu t=VROTL(a+rmd_fv<F>(b,c,d)+X[RMD_R1[j]]+vsplat(RMD_K1[F]),RMD_S1[j])+e;
        a=e; e=d; d=VROTL(c,10); c=b; b=t;
        t=VROTL(a2+rmd_fv<4-F>(b2,c2,d2)+X[RMD_R2[j]]+vsplat(RMD_K2[F]),RMD_S2[j])+e2;
        a2=e2; e2=d2; d2=VROTL(c2,10); c2=b2; b2=t;
    }
}
// hash160 de VLANES clés publiques compressées (xs[l], pars[l]) -> out[l] (même empaquetage que hash160_fast)
static void hash160_vec(const U256* xs,const u64* pars,U256* out){
    u32 Xl[8][VLANES], Pl[VLANES];
    for(int l=0;l<VLANES;l++){
        const U256& x=xs[l];
        Xl[0][l]=(u32)(x.d[3]>>32); Xl[1][l]=(u32)x.d[3]; Xl[2][l]=(u32)(x.d[2]>>32); Xl[3][l]=(u32)x.d[2];
        Xl[4][l]=(u32)(x.d[1]>>32); Xl[5][l]=(u32)x.d[1]; Xl[6][l]=(u32)(x.d[0]>>32); Xl[7][l]=(u32)x.d[0];
        Pl[l]=0x02u+(u32)pars[l];
    }
    vu X[8],w[64],H[8],rw[16];
    for(int i=0;i<8;i++) memcpy(&X[i],Xl[i],sizeof(vu));
    vu P; memcpy(&P,Pl,sizeof(vu));
    w[0]=(P<<24)|(X[0]>>8);
    for(int i=1;i<8;i++) w[i]=(X[i-1]<<24)|(X[i]>>8);
    w[8]=(X[7]<<24)|vsplat(0x00800000u);
    for(int i=9;i<15;i++) w[i]=vsplat(0);
    w[15]=vsplat(264);
    #pragma GCC unroll 64
    for(int i=16;i<64;i++){
        vu s0=VROTR(w[i-15],7)^VROTR(w[i-15],18)^(w[i-15]>>3);
        vu s1=VROTR(w[i-2],17)^VROTR(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    vu a=vsplat(0x6a09e667u),b=vsplat(0xbb67ae85u),c=vsplat(0x3c6ef372u),d=vsplat(0xa54ff53au);
    vu e=vsplat(0x510e527fu),f=vsplat(0x9b05688cu),g=vsplat(0x1f83d9abu),h=vsplat(0x5be0cd19u);
    #pragma GCC unroll 64
    for(int i=0;i<64;i++){
        vu S1=VROTR(e,6)^VROTR(e,11)^VROTR(e,25), ch=(e&f)^(~e&g);
        vu t1=h+S1+ch+vsplat(SHA_K[i])+w[i];
        vu S0=VROTR(a,2)^VROTR(a,13)^VROTR(a,22), mj=(a&b)^(a&c)^(b&c);
        vu t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    H[0]=a+vsplat(0x6a09e667u); H[1]=b+vsplat(0xbb67ae85u); H[2]=c+vsplat(0x3c6ef372u); H[3]=d+vsplat(0xa54ff53au);
    H[4]=e+vsplat(0x510e527fu); H[5]=f+vsplat(0x9b05688cu); H[6]=g+vsplat(0x1f83d9abu); H[7]=h+vsplat(0x5be0cd19u);
    for(int i=0;i<8;i++){
        vu q=H[i];
        rw[i]=(q>>24)|((q>>8)&vsplat(0x0000FF00u))|((q<<8)&vsplat(0x00FF0000u))|(q<<24);
    }
    rw[8]=vsplat(0x80); for(int i=9;i<14;i++) rw[i]=vsplat(0); rw[14]=vsplat(256); rw[15]=vsplat(0);
    vu a1=vsplat(0x67452301u),b1=vsplat(0xEFCDAB89u),c1=vsplat(0x98BADCFEu),d1=vsplat(0x10325476u),e1=vsplat(0xC3D2E1F0u);
    vu a2=a1,b2=b1,c2=c1,d2=d1,e2=e1;
    rmd_segv<0>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<1>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<2>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<3>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<4>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    vu o[5];
    o[0]=vsplat(0xEFCDAB89u)+c1+d2; o[1]=vsplat(0x98BADCFEu)+d1+e2; o[2]=vsplat(0x10325476u)+e1+a2;
    o[3]=vsplat(0xC3D2E1F0u)+a1+b2; o[4]=vsplat(0x67452301u)+b1+c2;
    u32 T[5][VLANES];
    for(int k=0;k<5;k++) memcpy(T[k],&o[k],sizeof(vu));
    for(int l=0;l<VLANES;l++){
        out[l].d[0]=T[0][l]|((u64)T[1][l]<<32); out[l].d[1]=T[2][l]|((u64)T[3][l]<<32);
        out[l].d[2]=T[4][l]; out[l].d[3]=0;
    }
}

// ---- Base58Check (P2PKH) et Bech32 (P2WPKH)
static const char* B58="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static std::string b58check_encode(unsigned char version,const unsigned char* h20){
    std::vector<unsigned char> d(25); d[0]=version; memcpy(&d[1],h20,20);
    unsigned char s1[32],s2[32]; sha256_bytes(d.data(),21,s1); sha256_bytes(s1,32,s2); memcpy(&d[21],s2,4);
    size_t zeros=0; while(zeros<d.size()&&d[zeros]==0) zeros++;
    std::vector<unsigned char> num(d.begin()+zeros,d.end()); std::string out;
    while(!num.empty()){
        std::vector<unsigned char> q; unsigned rem=0;
        for(unsigned char c:num){ unsigned cur=rem*256+c; unsigned qb=cur/58; rem=cur%58; if(!q.empty()||qb) q.push_back((unsigned char)qb); }
        out.push_back(B58[rem]); num=q;
    }
    out.append(zeros,'1'); std::reverse(out.begin(),out.end());
    return out;
}
// 0 = invalide, 1 = P2PKH valide (h20 rempli), 2 = valide mais non supporté (P2SH ...)
static int b58check_decode(const std::string& s,unsigned char* h20){
    if(s.empty()||s.size()>40) return 0;
    std::vector<unsigned char> num;                      // little-endian
    size_t zeros=0; while(zeros<s.size()&&s[zeros]=='1') zeros++;
    for(char ch:s){
        const char* p=strchr(B58,ch); if(!p||!ch) return 0;
        unsigned carry=(unsigned)(p-B58);
        for(size_t j=0;j<num.size();j++){ carry+=(unsigned)num[j]*58; num[j]=(unsigned char)carry; carry>>=8; }
        while(carry){ num.push_back((unsigned char)carry); carry>>=8; }
    }
    std::vector<unsigned char> d(zeros,0); d.insert(d.end(),num.rbegin(),num.rend());
    if(d.size()!=25) return 0;
    unsigned char s1[32],s2[32]; sha256_bytes(d.data(),21,s1); sha256_bytes(s1,32,s2);
    if(memcmp(s2,&d[21],4)!=0) return 0;
    if(d[0]!=0x00) return 2;
    memcpy(h20,&d[1],20); return 1;
}
static const char* B32="qpzry9x8gf2tvdw0s3jn54khce6mua7l";
static u32 bech32_polymod(const std::vector<unsigned char>& v){
    static const u32 G[5]={0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
    u32 chk=1;
    for(unsigned char x:v){ u32 b=chk>>25; chk=((chk&0x1ffffff)<<5)^x; for(int i=0;i<5;i++) if((b>>i)&1) chk^=G[i]; }
    return chk;
}
static std::vector<unsigned char> bech32_hrp_expand(const std::string& hrp){
    std::vector<unsigned char> v; for(char c:hrp) v.push_back((unsigned char)c>>5);
    v.push_back(0); for(char c:hrp) v.push_back((unsigned char)c&31); return v;
}
static std::string bech32_encode_p2wpkh(const unsigned char* h20){
    std::vector<unsigned char> data; data.push_back(0);
    int acc=0,bits=0;
    for(int i=0;i<20;i++){ acc=(acc<<8)|h20[i]; bits+=8; while(bits>=5){ bits-=5; data.push_back((acc>>bits)&31); } }
    if(bits) data.push_back((acc<<(5-bits))&31);
    std::vector<unsigned char> v=bech32_hrp_expand("bc"); v.insert(v.end(),data.begin(),data.end());
    for(int i=0;i<6;i++) v.push_back(0);
    u32 pm=bech32_polymod(v)^1;
    for(int i=0;i<6;i++) data.push_back((pm>>(5*(5-i)))&31);
    std::string out="bc1"; for(unsigned char x:data) out.push_back(B32[x]);
    return out;
}
static int bech32_decode_addr(const std::string& s0,unsigned char* h20){
    if(s0.size()<14||s0.size()>90) return 0;
    bool lo=false,up=false; for(char c:s0){ if(c>='a'&&c<='z') lo=true; if(c>='A'&&c<='Z') up=true; }
    if(lo&&up) return 0;
    std::string s=s0; for(auto& c:s) if(c>='A'&&c<='Z') c+=32;
    size_t pos=s.rfind('1'); if(pos==std::string::npos||pos<1||s.size()-pos<7) return 0;
    std::string hrp=s.substr(0,pos); std::vector<unsigned char> data;
    for(size_t i=pos+1;i<s.size();i++){ const char* p=strchr(B32,s[i]); if(!p||!s[i]) return 0; data.push_back((unsigned char)(p-B32)); }
    std::vector<unsigned char> v=bech32_hrp_expand(hrp); v.insert(v.end(),data.begin(),data.end());
    u32 pm=bech32_polymod(v);
    if(pm!=1&&pm!=0x2bc830a3) return 0;
    if(hrp!="bc") return 0;
    data.resize(data.size()-6);
    if(data.empty()) return 0;
    if(data[0]!=0||pm!=1) return 2;                      // taproot (v1+) ... : non supporté
    std::vector<unsigned char> prog; int acc=0,bits=0;
    for(size_t i=1;i<data.size();i++){ acc=(acc<<5)|data[i]; bits+=5; if(bits>=8){ bits-=8; prog.push_back((acc>>bits)&255); } }
    if(bits>=5||((acc<<(8-bits))&255)) return 0;
    if(prog.size()!=20) return 2;                        // P2WSH (32 octets) : non supporté
    memcpy(h20,prog.data(),20); return 1;
}
// ---------------------------------------------------------------- ensemble de cibles (Bloom + liste exacte)
struct Ent { U256 x; u64 par; };                       // X de la clé publique + parité de Y (0 = préfixe 02, 1 = 03)
static_assert(sizeof(Ent)==40,"taille Ent");
static inline bool x_less(const U256& a,const U256& b){
    for(int i=3;i>=0;--i){ if(a.d[i]!=b.d[i]) return a.d[i]<b.d[i]; }
    return false;
}
static bool ent_less(const Ent& a,const Ent& b){
    if(!eq(a.x,b.x)) return x_less(a.x,b.x);
    return a.par<b.par;
}
struct TargetSet {
    std::vector<Ent> list;                              // triée, sans doublon
    std::vector<u64> bloom;                             // nblocks * 8 mots de 64 bits (blocs de 512 bits)
    u64 nblocks=0;
    u64 src_hash=0;
    bool addr=false;                                    // true : la liste contient des hash160 (adresses)
};
// Filtre de Bloom "par blocs" : 1 bloc (une ligne de cache) par test, 6 bits par clé.
// X est uniforme : on lit directement ses bits, sans fonction de hachage.
static inline bool bloom_test(const u64* bl,u64 mask,const U256& x){
    const u64* blk=bl+((x.d[0]&mask)<<3);
    u64 h=x.d[1];
    for(int j=0;j<6;j++){
        u64 p=(h>>(9*j))&511;
        if(!((blk[p>>6]>>(p&63))&1)) return false;
    }
    return true;
}
static inline void bloom_insert(u64* bl,u64 mask,const U256& x){
    u64* blk=bl+((x.d[0]&mask)<<3);
    u64 h=x.d[1];
    for(int j=0;j<6;j++){ u64 p=(h>>(9*j))&511; blk[p>>6]|=1ULL<<(p&63); }
}
static void build_bloom(TargetSet& ts){
    u64 n=ts.list.size(), need=(n+15)/16; if(need<1) need=1;
    u64 nb=1; while(nb<need) nb<<=1;
    ts.nblocks=nb; ts.bloom.assign(nb*8,0);
    for(const Ent& e:ts.list) bloom_insert(ts.bloom.data(),nb-1,e.x);
}
static void finalize_targets(TargetSet& ts){
    std::sort(ts.list.begin(),ts.list.end(),ent_less);
    ts.list.erase(std::unique(ts.list.begin(),ts.list.end(),
        [](const Ent& a,const Ent& b){ return eq(a.x,b.x)&&a.par==b.par; }),ts.list.end());
    build_bloom(ts);
}
static bool ts_has_x(const TargetSet& ts,const U256& x){
    Ent probe; probe.x=x; probe.par=0;
    auto it=std::lower_bound(ts.list.begin(),ts.list.end(),probe,ent_less);
    return it!=ts.list.end() && eq(it->x,x);
}
static bool ts_has_exact(const TargetSet& ts,const U256& x,u64 par){
    Ent probe; probe.x=x; probe.par=par;
    return std::binary_search(ts.list.begin(),ts.list.end(),probe,ent_less);
}
static std::string ent_hex(const Ent& e){ return std::string(e.par?"03":"02")+to_hex(e.x); }
static std::string ent_hash_hex(const Ent& e){
    unsigned char b[20]; unpack20(e.x,b); char s[41]; for(int i=0;i<20;i++) snprintf(s+2*i,3,"%02x",b[i]); return s;
}
static std::string ent_addr_legacy(const Ent& e){ unsigned char b[20]; unpack20(e.x,b); return b58check_encode(0x00,b); }
static std::string ent_addr_bech32(const Ent& e){ unsigned char b[20]; unpack20(e.x,b); return bech32_encode_p2wpkh(b); }
static std::string ent_text(const TargetSet& ts,const Ent& e){ return ts.addr? ent_addr_legacy(e) : ent_hex(e); }
// Adresse -> Ent (hash160 empaqueté). Retour : 0 invalide, 1 ok, 2 valide mais non supportée (P2SH, P2WSH, Taproot)
static int parse_address(const std::string& s,Ent& e){
    unsigned char h[20];
    int r=b58check_decode(s,h);
    if(r==0) r=bech32_decode_addr(s,h);
    if(r==1){ e.x=pack20(h); e.par=0; }
    return r;
}

static bool parse_pub(const std::string& s,Ent& e){
    if(s.size()!=66||s[0]!='0'||(s[1]!='2'&&s[1]!='3')) return false;
    U256 x; if(!parse_hex(s.substr(2),x)||geq(x,PRIME)) return false;
    static U256 ex; static bool init=false;
    if(!init){ ex=PRIME; add_small(ex,1); ex=shr2(ex); init=true; }
    U256 r,y,y2; fe_sqr(r,x); fe_mul(r,r,x); U256 seven={{7,0,0,0}}; fe_add(r,r,seven);
    y=fe_pow(r,ex); fe_sqr(y2,y);
    if(!eq(y2,r)) return false;                          // X n'est pas sur la courbe
    e.x=x; e.par=(s[1]=='3'); return true;
}
static void parse_targets_text(const std::string& txt,TargetSet& ts,bool addr,u64& lines,u64& rejected,u64& dups,u64& unsupported){
    lines=rejected=dups=unsupported=0; ts.addr=addr;
    size_t i=0;
    if(txt.size()>=3&&(unsigned char)txt[0]==0xEF&&(unsigned char)txt[1]==0xBB&&(unsigned char)txt[2]==0xBF) i=3;
    while(i<txt.size()){
        size_t j=txt.find('\n',i); if(j==std::string::npos) j=txt.size();
        size_t a=i,b=j;
        while(a<b&&(txt[a]==' '||txt[a]=='\t'||txt[a]=='\r')) a++;
        size_t t=a; while(t<b&&txt[t]!=' '&&txt[t]!='\t'&&txt[t]!='\r') t++;   // 1er mot de la ligne
        if(t>a){
            lines++;
            Ent e; std::string tok=txt.substr(a,t-a);
            if(addr){
                int rc=parse_address(tok,e);
                if(rc==1) ts.list.push_back(e); else if(rc==2) unsupported++; else rejected++;
            } else if(parse_pub(tok,e)) ts.list.push_back(e); else rejected++;
        }
        i=j+1;
    }
    u64 valid=ts.list.size();
    finalize_targets(ts);
    dups=valid-ts.list.size();
}
static bool read_file(const std::string& path,std::string& out){
    FILE* f=fopen(path.c_str(),"rb"); if(!f) return false;
    out.clear(); char buf[1<<16]; size_t n;
    while((n=fread(buf,1,sizeof buf,f))>0) out.append(buf,n);
    fclose(f); return true;
}
static u64 fnv64(const std::string& s){
    u64 h=1469598103934665603ULL;
    for(unsigned char c:s){ h^=c; h*=1099511628211ULL; }
    return h;
}
static u64 sum_hash(const void* p,size_t n,u64 h){
    const unsigned char* c=(const unsigned char*)p; size_t i=0;
    for(;i+8<=n;i+=8){ u64 w; memcpy(&w,c+i,8); h=(h^w)*0x9E3779B97F4A7C15ULL; h^=h>>29; }
    for(;i<n;i++){ h=(h^c[i])*0x100000001B3ULL; }
    return h;
}

// Fichier .bloom : en-tête (6 x u64) + filtre + liste exacte
static const char MAGIC[8]={'P','Z','B','L','O','O','M','2'};
static const u64 ENDIAN_TAG=0x0102030405060708ULL;
static u64 payload_sum(const TargetSet& ts){
    u64 h=0x1234567812345678ULL;
    h=sum_hash(ts.bloom.data(),ts.bloom.size()*8,h);
    h=sum_hash(ts.list.data(),ts.list.size()*sizeof(Ent),h);
    return h;
}
static bool save_bloom(const TargetSet& ts,const std::string& path){
    FILE* f=fopen(path.c_str(),"wb"); if(!f) return false;
    u64 hdr[6]={ENDIAN_TAG,(u64)ts.list.size(),ts.nblocks,ts.src_hash,payload_sum(ts),(u64)(ts.addr?1:0)};
    bool ok=fwrite(MAGIC,1,8,f)==8 && fwrite(hdr,8,6,f)==6
        && fwrite(ts.bloom.data(),8,ts.bloom.size(),f)==ts.bloom.size()
        && fwrite(ts.list.data(),sizeof(Ent),ts.list.size(),f)==ts.list.size();
    ok=(fclose(f)==0)&&ok;
    return ok;
}
static bool load_bloom(const std::string& path,TargetSet& ts,std::string& err){
    FILE* f=fopen(path.c_str(),"rb"); if(!f){ err="fichier introuvable"; return false; }
    char mg[8]; u64 hdr[6];
    if(fread(mg,1,8,f)!=8||memcmp(mg,MAGIC,8)!=0||fread(hdr,8,6,f)!=6){ fclose(f); err="format invalide"; return false; }
    if(hdr[0]!=ENDIAN_TAG){ fclose(f); err="format d'octets incompatible"; return false; }
    u64 n=hdr[1], nb=hdr[2];
    if(n==0||n>(1ULL<<32)||nb==0||(nb&(nb-1))||nb>(1ULL<<32)){ fclose(f); err="en-tête incohérent"; return false; }
    ts.nblocks=nb; ts.src_hash=hdr[3]; ts.addr=(hdr[5]==1);
    ts.bloom.assign(nb*8,0); ts.list.assign(n,Ent());
    bool ok=fread(ts.bloom.data(),8,ts.bloom.size(),f)==ts.bloom.size()
         && fread(ts.list.data(),sizeof(Ent),n,f)==n;
    char extra; if(ok && fread(&extra,1,1,f)==1) ok=false;          // fichier trop long
    fclose(f);
    if(!ok){ err="fichier tronqué ou corrompu"; return false; }
    if(payload_sum(ts)!=hdr[4]){ err="somme de contrôle invalide"; return false; }
    for(size_t i=0;i<n;i++){
        if(i&&!ent_less(ts.list[i-1],ts.list[i])){ err="liste non triée"; return false; }
        if(!bloom_test(ts.bloom.data(),nb-1,ts.list[i].x)){ err="filtre incohérent"; return false; }
    }
    return true;
}

// ---------------------------------------------------------------- recherche : état partagé
static const int HALF=512;                 // chaque lot teste 2*HALF clés consécutives autour d'un point central
static std::vector<U256> gx,gy;            // i*G pour i = 1..HALF
static Aff SST;                            // (2*HALF)*G : pas vers le lot suivant

static void build_tables(){
    gx.assign(HALF+1,ZERO); gy.assign(HALF+1,ZERO);
    Aff cur=GEN;
    for(int i=1;i<=HALF;i++){ gx[i]=cur.x; gy[i]=cur.y; if(i<HALF) cur=aff_add(cur,GEN); }
    SST=aff_add(cur,cur);                  // cur = HALF*G
}

// ---- Tables pour le GPU : i*G (i <= gpu_half), pas 2*gpu_half*G, et table de multiplication à base fixe
// (64 fenêtres de 4 bits : point d*16^w*G pour d = 1..15) qui permet à chaque thread GPU de calculer k*G seul.
struct GpuTables {
    int half=0; bool ready=false;
    std::vector<U256> gx,gy; U256 sst[2];
    std::vector<uint32_t> ftab;
};
static GpuTables GT;
static void build_gpu_tables(){
    if(GT.ready) return;
    int h=gpu_half(); GT.half=h;
    GT.gx.assign(h+1,ZERO); GT.gy.assign(h+1,ZERO);
    Aff cur=GEN;
    for(int i=1;i<=h;i++){ GT.gx[i]=cur.x; GT.gy[i]=cur.y; if(i<h) cur=aff_add(cur,GEN); }
    Aff s=aff_add(cur,cur);                                  // 2*h*G
    GT.sst[0]=s.x; GT.sst[1]=s.y;
    GT.ftab.assign(64*15*16,0);
    Aff base=GEN;
    for(int w=0;w<64;w++){
        Aff c=base;
        for(int d=1;d<=15;d++){
            uint32_t* e=&GT.ftab[(size_t)(w*15+d-1)*16];
            memcpy(e,&c.x,32); memcpy(e+8,&c.y,32);
            if(d<15) c=aff_add(c,base);
        }
        base=aff_add(c,base);                                // 16 * base
    }
    GT.ready=true;
}

struct Found { U256 key; Ent ent; std::string pub; double t=0; bool saved=false; bool outside=false; std::string path; };
struct TState {
    std::atomic<u64> keys; std::atomic<u64> block; std::atomic<int> mode;
    char pad[40];
    TState():keys(0),block(0),mode(-1){}
};
struct Work { U256 s,e; };                 // intervalle de clés [s,e] (inclus)
typedef std::chrono::steady_clock clk;

// Modes de parcours des blocs (un seul par lancement, tous les cœurs travaillent sur la même suite)
enum Mode { MD_SOBOL=0, MD_RANDOM=1, MD_SEQ=2, MD_MC=3, MD_PI=4, MD_E=5, MD_LN=6, MD_COUNT=7 };
static const char* MODE_NAME[MD_COUNT]={"Sobol","Aléatoire","Séquentiel","Monte Carlo","Pi","e","ln 2"};
// parties fractionnaires de pi, e et ln(2) sur 64 bits (suites de Weyl : bloc = frac(i*c) * nombre_de_blocs)
static const u64 WEYL_C[3]={0x243F6A8885A308D3ULL,0xB7E151628AED2A6BULL,0xB17217F7D1CF79ABULL};

static long double u256_to_ld(const U256& a){
    long double r=0, m=1; const long double B=18446744073709551616.0L;
    for(int i=0;i<4;i++){ r+=(long double)a.d[i]*m; m*=B; }
    return r;
}
// Ordre n de la courbe secp256k1 : toute clé privée valide est dans [1, n-1]
static const U256 CURVE_N={{0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL}};

struct Run {
    const TargetSet* ts=nullptr;
    int mode=MD_RANDOM;
    std::atomic<u64> seq;                                // compteur partagé (modes séquentiel, Sobol, Pi, e, ln)
    int bits=0,nthreads=1,bsize=0,nbl=0;
    U256 start,end,blockmask;
    u64 NB=1, NBP=1;                                     // NB = nombre de blocs ; NBP = puissance de 2 >= NB (suites Sobol, Pi, ...)
    bool is_range=false; long double total=1;            // plage personnalisée ; nombre total de clés
    std::vector<std::atomic<u64>> claim;                // blocs déjà pris par un cœur
    std::vector<TState> st;
    std::atomic<bool> stop; std::atomic<int> done; std::atomic<int> nfound;
    SpinLock lock; std::vector<Found> found;
    bool write_file=false, emergency=true, quiet=false;
    std::string out_main="cle_trouvee.txt", out_copy="cle_trouvee_copie.txt";   // deux copies du résultat
    clk::time_point t0;
    // ---- calculateurs : un par carte graphique (gdev >= 0) ou par cœur CPU (gdev = -1)
    std::vector<int> gdev; std::vector<std::string> label;
    // ---- reprise : travail non terminé de chaque calculateur (wip) + intervalles hérités d'une session précédente (pending).
    // ck protège ensemble "prise d'un bloc + inscription dans wip" pour que les sauvegardes soient cohérentes.
    SpinLock ck; std::vector<std::vector<Work>> wip; std::vector<Work> pending;
    std::string ckpt_path; double ckpt_every=60; u64 ckpt_hash=0;
    double base_elapsed=0; u64 base_tested=0;           // cumul des sessions précédentes
    double auto_stop_s=-1;                              // (tests) arrêt automatique après N secondes
    Run():seq(0),stop(false),done(0),nfound(0){}
};

// Plage quelconque [start,end] (1 <= start <= end < n)
static void init_run_range(Run& r,const U256& start,const U256& end,int nthreads,const TargetSet* ts,bool write_file,int mode=MD_RANDOM,bool is_range=true){
    r.ts=ts; r.nthreads=nthreads; r.write_file=write_file; r.mode=mode; r.seq.store(0);
    r.start=start; r.end=end; r.is_range=is_range; r.bits=highest_bit(end)+1;
    U256 sz; sub_raw(sz,end,start);                     // taille - 1
    r.total=u256_to_ld(sz)+1;
    int L=highest_bit(sz)+1; if(L<0) L=0;               // taille <= 2^L
    int b=(L>=7)? std::min(20,L-6) : 0;
    if(L-b>26) b=L-26;                                  // au plus 2^26 blocs
    r.bsize=b; r.blockmask=low_mask(b);
    U256 t=sz; for(int i=0;i<b;i++){ for(int k=0;k<4;k++){ t.d[k]>>=1; if(k<3) t.d[k]|=t.d[k+1]<<63; } }
    r.NB=t.d[0]+1;                                      // = ceil(taille / 2^b)  (<= 2^26)
    r.nbl=0; while((1ULL<<r.nbl)<r.NB) r.nbl++;
    r.NBP=1ULL<<r.nbl;
    u64 nw=(r.NBP+63)/64;
    r.claim=std::vector<std::atomic<u64>>(nw);
    for(u64 i=0;i<nw;i++) r.claim[i].store(0);
    for(u64 bb=r.NB;bb<nw*64;bb++) r.claim[bb>>6].fetch_or(1ULL<<(bb&63));   // blocs hors plage = déjà pris
    r.st=std::vector<TState>(nthreads);
    r.gdev.assign(nthreads,-1);
    r.label.clear(); for(int i=0;i<nthreads;i++){ char b[32]; snprintf(b,sizeof b,"CPU %d",i+1); r.label.push_back(b); }
    r.wip.assign(nthreads,std::vector<Work>()); r.pending.clear();
    r.found.clear(); r.stop=false; r.done=0; r.nfound=0;
}
// Réaffecte les calculateurs d'un Run déjà initialisé : gpus = indices des cartes, ncpu = nombre de cœurs CPU en plus.
static void set_workers(Run& r,const std::vector<int>& gpus,int ncpu){
    int n=(int)gpus.size()+ncpu;
    r.nthreads=n; r.st=std::vector<TState>(n);
    r.gdev.assign(n,-1); r.label.assign(n,"");
    r.wip.assign(n,std::vector<Work>());
    for(size_t i=0;i<gpus.size();i++){
        r.gdev[i]=gpus[i]; GpuInfo gi; std::string nm="GPU";
        char b[160]; if(gpu_device_info(gpus[i],&gi)){ snprintf(b,sizeof b,"GPU%d %s",gpus[i],gi.name); nm=b; }
        else { snprintf(b,sizeof b,"GPU%d",gpus[i]); nm=b; }
        r.label[i]=nm;
    }
    for(int i=0;i<ncpu;i++){ char b[32]; snprintf(b,sizeof b,"CPU %d",i+1); r.label[gpus.size()+i]=b; }
}
// Plage d'un nombre de bits : [2^(bits-1), 2^bits - 1]
static void init_run(Run& r,int bits,int nthreads,const TargetSet* ts,bool write_file,int mode=MD_RANDOM){
    U256 st=ZERO; st.d[(bits-1)>>6]|=1ULL<<((bits-1)&63);
    init_run_range(r,st,low_mask(bits),nthreads,ts,write_file,mode,false);
}

static inline bool claim_block(Run& r,u64 b){
    std::atomic<u64>& w=r.claim[b>>6]; u64 m=1ULL<<(b&63);
    if(w.load(std::memory_order_relaxed)&m) return false;
    return !(w.fetch_or(m)&m);
}
static inline u64 rev_bits(u64 v,int nb){
    u64 r=0; for(int i=0;i<nb;i++){ r=(r<<1)|(v&1); v>>=1; } return r;
}

// Générateurs d'ordre de blocs. Chaque bloc n'est donné qu'une seule fois (liste partagée "claim").
struct Gen {
    Run* r; int type; Rng rng;
    u64 ZS=1,hits=0; size_t pos=0;
    std::vector<u32> order;
    Gen(Run* rr,int t,int tid):r(rr),type(t){
        rng.seed((u64)clk::now().time_since_epoch().count()*0x9E3779B97F4A7C15ULL+(u64)tid*7919+(u64)t*104729);
        if(type==MD_MC){
            u64 NBp=r->NBP, Z=std::min<u64>(NBp,4096); ZS=NBp/Z;
            order.resize(Z); std::iota(order.begin(),order.end(),0u);
            for(size_t i=order.size();i>1;--i) std::swap(order[i-1],order[rng.next()%i]);
        }
    }
    // Balayage final : prend un bloc encore libre (garantit que toute la plage est couverte, sans bloc refait)
    bool sweep(u64& blk,u64 w0){
        u64 nw=r->claim.size();
        for(u64 j=0;j<nw;j++){
            u64 w=(w0+j)%nw; u64 cur=r->claim[w].load();
            while(~cur){
                u64 b=w*64+(u64)__builtin_ctzll(~cur);
                if(claim_block(*r,b)){ blk=b; return true; }
                cur=r->claim[w].load();
            }
        }
        return false;
    }
    bool next(u64& blk){
        u64 NB=r->NB;
        switch(type){
        case MD_SEQ:                                       // du début à la fin de la plage
            for(;;){
                u64 n=r->seq.fetch_add(1); if(n>=NB) return false;
                if(claim_block(*r,n)){ blk=n; return true; }
            }
        case MD_SOBOL:                                     // suite de Sobol (dimension 1, ordre de Gray) : permutation des blocs
            for(;;){
                u64 n=r->seq.fetch_add(1); if(n>=r->NBP) return false;
                u64 b=rev_bits(n^(n>>1),r->nbl);
                if(claim_block(*r,b)){ blk=b; return true; }
            }
        case MD_PI: case MD_E: case MD_LN: {                 // suite de Weyl : bloc = frac(i * c) * NB, puis balayage final
            const u64 C=WEYL_C[type-MD_PI], limit=4*r->NBP+64;
            for(;;){
                u64 i=r->seq.fetch_add(1);
                if(i>=limit) return sweep(blk,0);
                u64 b=r->nbl? ((i*C)>>(64-r->nbl)) : 0;
                if(claim_block(*r,b)){ blk=b; return true; }
            }
        }
        case MD_RANDOM:                                    // blocs tirés au hasard, sans répétition
            for(int t=0;t<16;t++){ u64 b=rng.next()&(r->NBP-1); if(claim_block(*r,b)){ blk=b; return true; } }
            return sweep(blk,rng.next()%r->claim.size());
        default: break;
        }
        for(;;){                                          // Monte Carlo : zones égales visitées dans un ordre aléatoire
            if(pos>=order.size()){
                if(pos>0&&hits==0) return false;
                for(size_t i=order.size();i>1;--i) std::swap(order[i-1],order[rng.next()%i]);
                pos=0; hits=0;
            }
            u64 base=(u64)order[pos++]*ZS;
            for(int t=0;t<8;t++){ u64 b=base+(rng.next()&(ZS-1)); if(claim_block(*r,b)){ hits++; blk=b; return true; } }
            u64 off=rng.next()&(ZS-1);
            for(u64 j=0;j<ZS;j++){ u64 b=base+((off+j)&(ZS-1)); if(claim_block(*r,b)){ hits++; blk=b; return true; } }
        }
    }
};

// ---------------------------------------------------------------- résultat trouvé
static Ent ent_of_point(bool addr,const Aff& P){
    Ent e;
    if(addr){ e.x=hash160_point_bytes(P); e.par=0; }
    else { e.x=P.x; e.par=P.y.d[0]&1; }
    return e;
}
// ---------------------------------------------------------------- enregistrement fiable des résultats
static std::string result_text(const Run& r,const Found& f){
    std::string t; char b[96];
    if(r.ts->addr){
        t+="Adresse (P2PKH)     : "+ent_addr_legacy(f.ent)+"\n";
        t+="Adresse (bech32)    : "+ent_addr_bech32(f.ent)+"\n";
        t+="Hash160             : "+ent_hash_hex(f.ent)+"\n";
    }
    t+="Clé publique        : "+f.pub+"\n";
    if(r.is_range) t+="Plage               : "+to_hex(r.start)+" -> "+to_hex(r.end)+"\n";
    else { snprintf(b,sizeof b,"Bits                : %d\n",r.bits); t+=b; }
    if(f.outside) t+="Hors plage demandée : OUI (clé valide, trouvée juste après la fin de la plage)\n";
    t+="Clé privée (hex)    : "+to_hex(f.key)+"\n";
    t+="Clé privée (décimal): "+to_dec(f.key)+"\n";
    t+="Vérification        : VALIDE\n";
    snprintf(b,sizeof b,"Temps (s)           : %.1f\n",f.t); t+=b;
    t+="----------------------------------------\n";
    return t;
}
// Ajout en fin de fichier, vidé sur le disque (flush + fsync)
static bool append_result(const std::string& path,const std::string& text){
    FILE* f=fopen(path.c_str(),"ab"); if(!f) return false;
    bool ok=fwrite(text.data(),1,text.size(),f)==text.size();
    if(fflush(f)!=0) ok=false;
#ifdef _WIN32
    if(_commit(_fileno(f))!=0) ok=false;
#else
    if(fsync(fileno(f))!=0) ok=false;
#endif
    if(fclose(f)!=0) ok=false;
    return ok;
}
// Écrit puis RELIT le fichier : le texte doit s'y trouver à la fin, sinon l'enregistrement est considéré comme raté
static bool save_checked(const std::string& path,const std::string& text){
    if(!append_result(path,text)) return false;
    std::string c; if(!read_file(path,c)) return false;
    return c.size()>=text.size() && c.compare(c.size()-text.size(),text.size(),text)==0;
}
static void record_find(Run& r,const U256& key,const Aff& P){
    r.lock.lock();
    for(const Found& f:r.found) if(eq(f.key,key)){ r.lock.unlock(); return; }
    Found f; f.key=key; f.ent=ent_of_point(r.ts->addr,P); f.pub=compress_hex(P);
    f.t=std::chrono::duration<double>(clk::now()-r.t0).count();
    f.outside=!geq(r.end,key)||!geq(key,r.start);
    f.saved=!r.write_file;
    if(r.write_file){
        std::string text=result_text(r,f), used; int ok=0;
        const std::string* names[2]={&r.out_main,&r.out_copy};
        for(int i=0;i<2;i++){
            if(names[i]->empty()) continue;
            if(save_checked(*names[i],text)){ ok++; if(!used.empty()) used+=" + "; used+=*names[i]; }
        }
        if(!ok&&r.emergency){                                  // secours : autres emplacements
            std::vector<std::string> alt;
            alt.push_back("cle_trouvee_"+std::to_string((long long)time(nullptr))+".txt");
            for(const char* ev:{"TEMP","TMP","TMPDIR","USERPROFILE","HOME"}) if(const char* d=getenv(ev)) alt.push_back(std::string(d)+"/cle_trouvee_urgence.txt");
            alt.push_back("/tmp/cle_trouvee_urgence.txt");
            for(const std::string& a2:alt) if(save_checked(a2,text)){ ok++; used=a2; break; }
        }
        f.saved=ok>0; f.path=used;
        if(!f.saved&&!r.quiet){
            fprintf(stderr,"\n\n*** ATTENTION : clé trouvée mais IMPOSSIBLE à enregistrer ! Note-la maintenant : ***\n%s\n",text.c_str());
            fflush(stderr);
        }
    }
    r.found.push_back(f);
    r.nfound.fetch_add(1);
    if(r.found.size()>=r.ts->list.size()) r.stop.store(true);
    r.lock.unlock();
}
// Confirmation indépendante : k*G recalculé en arithmétique affine (et hash160 par le chemin "octets"),
// puis comparaison exacte avec la liste des cibles.
static void verify_point(Run& r,const U256& k){
    if(is_zero(k)) return;
    Aff P=aff_mul(k,GEN);
    if(P.inf) return;
    bool hit=r.ts->addr ? ts_has_exact(*r.ts,hash160_point_bytes(P),0)
                        : ts_has_exact(*r.ts,P.x,P.y.d[0]&1);
    if(hit) record_find(r,k,P);
}

struct Scratch {
    std::vector<U256> pre,cx; std::vector<char> zero; std::vector<u64> cpar; std::vector<int> coff;
    Scratch():pre(HALF+2),cx(2*HALF+2*VLANES+2),zero(HALF+2),cpar(2*HALF+2*VLANES+2),coff(2*HALF+2*VLANES+2){}
};

// ---- Distribution du travail. Prendre un intervalle et l'inscrire dans wip[tid] se fait sous le même verrou :
// une sauvegarde de reprise voit donc toujours chaque bloc pris soit "non commencé", soit dans le travail d'un calculateur.
static bool take_work(Run& r,Gen& g,int tid,U256& s,U256& e){
    r.ck.lock();
    bool ok=false;
    if(!r.pending.empty()){ Work w=r.pending.back(); r.pending.pop_back(); s=w.s; e=w.e; ok=true; }
    else {
        u64 blk;
        if(g.next(blk)){
            s=r.start; add_raw(s,s,shl_u64(blk,r.bsize));
            e=s; add_raw(e,s,r.blockmask);
            if(!geq(r.end,e)) e=r.end;                                       // dernier bloc : s'arrête à la fin de la plage
            ok=true;
        }
    }
    if(ok&&tid>=0&&tid<(int)r.wip.size()) r.wip[tid].push_back(Work{s,e});
    r.ck.unlock();
    return ok;
}
static void wip_set(Run& r,int tid,const std::vector<Work>& w){
    if(tid<0||tid>=(int)r.wip.size()) return;
    r.ck.lock(); r.wip[tid]=w; r.ck.unlock();
}
static void wip_set_one(Run& r,int tid,const U256& s,const U256& e){
    if(tid<0||tid>=(int)r.wip.size()) return;
    r.ck.lock(); r.wip[tid].assign(1,Work{s,e}); r.ck.unlock();
}
// numéro de bloc contenant la clé s (pour l'affichage)
static u64 block_index(const Run& r,const U256& s){
    U256 d; sub_raw(d,s,r.start);
    for(int i=0;i<r.bsize;i++){ for(int k=0;k<4;k++){ d.d[k]>>=1; if(k<3) d.d[k]|=d.d[k+1]<<63; } }
    return d.d[0];
}

// Parcourt [start,end] (un bloc) par lots de 2*HALF clés consécutives.
// Un point central C sert aux clés kc-i et kc+i : une seule inversion (par lots) pour deux clés.
template<bool ADDR>
static void search_block(Run& r,const U256& start,const U256& end,Scratch& sc,TState& st,int tid=-1){
    const TargetSet& ts=*r.ts;
    const u64* bl=ts.bloom.data(); const u64 mask=ts.nblocks-1;
    U256 lo=start, kc=start; add_small(kc,(u64)HALF);          // clés du lot : kc-HALF .. kc+HALF-1
    Aff C=jac_to_aff(jac_mul(kc,GEN));
    u64 pubcnt=0;
    for(;;){
        if(r.stop.load(std::memory_order_relaxed)){ wip_set_one(r,tid,lo,end); return; }   // reste à faire : [lo,end]
        if(tid>=0&&((++pubcnt)&0xFFF)==0) wip_set_one(r,tid,lo,end);                       // ~ toutes les 4 M de clés
        U256 diff; sub_raw(diff,end,lo);
        bool last=((diff.d[1]|diff.d[2]|diff.d[3])==0) && diff.d[0]<(u64)(2*HALF);
        u64 count=last? diff.d[0]+1 : (u64)(2*HALF);
        int nc=0;
        auto key_of=[&](int off){ U256 k=kc; if(off>=0) add_small(k,(u64)off); else { U256 m={{(u64)(-off),0,0,0}}; sub_raw(k,kc,m); } return k; };
        auto emit=[&](const U256& x,u64 par,int off){
            if(ADDR){ sc.cx[nc]=x; sc.cpar[nc]=par; sc.coff[nc]=off; nc++; }
            else if(bloom_test(bl,mask,x) && ts_has_x(ts,x)) verify_point(r,key_of(off));
        };
        emit(C.x,C.y.d[0]&1,0);                                   // clé centrale

        // inversion par lots de (gx[i]-C.x), i = 1..HALF, puis du pas vers le lot suivant
        U256 acc=ONE;
        for(int i=1;i<=HALF;i++){
            U256 d; fe_sub(d,gx[i],C.x);
            if(is_zero(d)){ sc.zero[i]=1; d=ONE; } else sc.zero[i]=0;
            fe_mul(acc,acc,d); sc.pre[i]=acc;
        }
        U256 dS; fe_sub(dS,SST.x,C.x);
        bool zs=is_zero(dS); if(zs) dS=ONE;
        fe_mul(acc,acc,dS);
        U256 inv=fe_inv(acc);
        Aff Cn; Cn.inf=false; Cn.x=ZERO; Cn.y=ZERO;
        {
            U256 invd; fe_mul(invd,inv,sc.pre[HALF]); fe_mul(inv,inv,dS);
            if(!zs&&!last){
                U256 t,l; fe_sub(t,SST.y,C.y); fe_mul(l,t,invd);
                fe_sqr(Cn.x,l); fe_sub(Cn.x,Cn.x,C.x); fe_sub(Cn.x,Cn.x,SST.x);
                fe_sub(t,C.x,Cn.x); fe_mul(Cn.y,l,t); fe_sub(Cn.y,Cn.y,C.y);
            }
        }
        for(int i=HALF;i>=1;--i){
            U256 invd;
            if(i>1){
                fe_mul(invd,inv,sc.pre[i-1]);
                U256 d; fe_sub(d,gx[i],C.x); if(sc.zero[i]) d=ONE;
                fe_mul(inv,inv,d);
            } else invd=inv;
            if(sc.zero[i]){                                       // cas dégénéré (C = ±i*G) : chemin lent sûr
                if(i<HALF) verify_point(r,key_of(i));
                verify_point(r,key_of(-i));
                continue;
            }
            U256 t,l,x3,y3;
            // clé kc-i : C + (-iG)
            fe_add(t,gy[i],C.y); fe_mul(l,t,invd);
            fe_sqr(x3,l); fe_sub(x3,x3,C.x); fe_sub(x3,x3,gx[i]);
            u64 par=0;
            if(ADDR){ fe_sub(t,x3,C.x); fe_mul(y3,l,t); fe_sub(y3,y3,C.y); par=y3.d[0]&1; }
            emit(x3,par,-i);
            if(i<HALF){                                           // clé kc+i : C + iG
                fe_sub(t,gy[i],C.y); fe_mul(l,t,invd);
                fe_sqr(x3,l); fe_sub(x3,x3,C.x); fe_sub(x3,x3,gx[i]);
                if(ADDR){ fe_sub(t,C.x,x3); fe_mul(y3,l,t); fe_sub(y3,y3,C.y); par=y3.d[0]&1; }
                emit(x3,par,i);
            }
        }
        if(ADDR){                                                 // hash160 par paquets de VLANES clés
            for(int c=0;c<nc;c+=VLANES){
                U256 xs[VLANES],hs[VLANES]; u64 ps[VLANES]; int m=std::min(VLANES,nc-c);
                for(int l=0;l<VLANES;l++){ int idx=c+(l<m?l:m-1); xs[l]=sc.cx[idx]; ps[l]=sc.cpar[idx]; }
                hash160_vec(xs,ps,hs);
                for(int l=0;l<m;l++)
                    if(bloom_test(bl,mask,hs[l]) && ts_has_x(ts,hs[l])) verify_point(r,key_of(sc.coff[c+l]));
            }
        }
        st.keys.fetch_add(count,std::memory_order_relaxed);
        if(last) break;
        add_small(lo,(u64)(2*HALF)); add_small(kc,(u64)(2*HALF));
        if(zs) C=jac_to_aff(jac_mul(kc,GEN)); else C=Cn;
    }
}

struct WArg { Run* r; int tid; };

static void cpu_range(Run& r,const U256& s,const U256& e,Scratch& sc,TState& st,int tid){
    if(r.ts->addr) search_block<true>(r,s,e,sc,st,tid); else search_block<false>(r,s,e,sc,st,tid);
}
// Boucle d'un cœur CPU : prend des blocs jusqu'à épuisement de la plage
static void cpu_loop(Run& r,Gen& g,int tid,TState& st,Scratch& sc){
    while(!r.stop.load()){
        U256 s,e;
        if(!take_work(r,g,tid,s,e)) break;                                   // tous les blocs sont pris
        st.mode.store(g.type); st.block.store(block_index(r,s));
        cpu_range(r,s,e,sc,st,tid);
        if(!r.stop.load()) wip_set(r,tid,std::vector<Work>());               // bloc terminé
    }
}

// ---- Calculateur GPU : une carte. Les blocs sont découpés en "travaux" (un thread GPU chacun : g_gpu_jb lots de
// 2*gpu_half() clés consécutives). Le GPU renvoie les candidats qui passent le filtre de Bloom ; l'hôte les confirme
// (liste exacte, puis recalcul indépendant en arithmétique affine). Les travaux que le GPU déclare dégénérés
// (clé proche de 0 ou de l'ordre de la courbe : plages minuscules) sont refaits sur CPU.
static int g_gpu_jb=16;
static void gpu_worker(Run& r,int tid,Gen& g,TState& st,Scratch& sc){
    const int dev=r.gdev[tid];
    const int GH=gpu_half(), GS=2*GH, JB=g_gpu_jb;
    const TargetSet& ts=*r.ts;
    GpuInfo gi; if(!gpu_device_info(dev,&gi)){ memset(&gi,0,sizeof gi); gi.sm_count=8; }
    char err[512]; err[0]=0;
    uint32_t maxjobs=(uint32_t)std::max(4096,std::min(1<<20,gi.sm_count*2048));
    const uint32_t hitcap=1u<<18;
    GpuCtx* ctx=gpu_open(dev,(const uint32_t*)GT.gx.data(),(const uint32_t*)GT.gy.data(),(const uint32_t*)GT.sst,
                         GT.ftab.data(),ts.bloom.data(),(uint64_t)ts.bloom.size(),ts.nblocks-1,ts.addr?1:0,maxjobs,hitcap,err,sizeof err);
    if(!ctx){
        if(!r.quiet) fprintf(stderr,"\n[%s] GPU inutilisable (%s) : ce calculateur continue sur CPU.\n",r.label[tid].c_str(),err);
        cpu_loop(r,g,tid,st,sc); return;
    }
    TState scratch_state;                                                       // compteur factice pour les reprises CPU
    std::vector<GJob> jobs(maxjobs); std::vector<u64> jkeys(maxjobs);
    std::vector<GHit> hits(hitcap); std::vector<uint32_t> bad(maxjobs);
    U256 cur=ZERO,bend=ZERO; bool have=false;
    uint32_t J=std::min<uint32_t>(maxjobs,std::max<uint32_t>(256,(uint32_t)gi.sm_count*64u));
    bool gpu_fail=false;

    // Confirme les candidats du travail [first, first+cnt) et refait les travaux dégénérés. false = erreur GPU.
    std::function<bool(uint32_t,uint32_t)> run_range=[&](uint32_t first,uint32_t cnt)->bool{
        uint32_t nh=0,nbad=0;
        if(gpu_run(ctx,jobs.data()+first,cnt,hits.data(),&nh,bad.data(),&nbad,err,sizeof err)!=0) return false;
        if(nh>hitcap){                                                          // trop de candidats : on relance en deux morceaux
            if(cnt>1){ uint32_t h=cnt/2; return run_range(first,h)&&run_range(first+h,cnt-h); }
            // un seul travail ne peut pas dépasser hitcap candidats (JB*GS << hitcap) ; par prudence : refait sur CPU
            nbad=1; bad[0]=0; nh=0;
        }
        for(uint32_t k=0;k<nh;k++){
            const GHit& h=hits[k]; const GJob& jb=jobs[first+h.job];
            U256 key; memcpy(&key,jb.lo,32); add_small(key,(u64)GH);
            if(h.off>=0) add_small(key,(u64)h.off); else { U256 m={{(u64)(-(int64_t)h.off),0,0,0}}; sub_raw(key,key,m); }
            U256 x=ZERO;
            x.d[0]=h.w[0]|((u64)h.w[1]<<32); x.d[1]=h.w[2]|((u64)h.w[3]<<32);
            x.d[2]=h.w[4]|((u64)h.w[5]<<32); x.d[3]=h.w[6]|((u64)h.w[7]<<32);
            if(ts_has_x(ts,x)) verify_point(r,key);                             // filtre exact, puis recalcul indépendant
        }
        for(uint32_t k=0;k<nbad;k++){
            const GJob& jb=jobs[first+bad[k]];
            U256 s; memcpy(&s,jb.lo,32); U256 e=s; add_small(e,(u64)jb.nb*(u64)GS-1);
            cpu_range(r,s,e,sc,scratch_state,-1);
        }
        return true;
    };

    while(!r.stop.load()){
        // 1) assembler les travaux de ce lancement
        uint32_t nj=0;
        while(nj<J){
            if(!have){
                if(!take_work(r,g,tid,cur,bend)) break;
                have=true; st.mode.store(g.type); st.block.store(block_index(r,cur));
            }
            U256 diff; sub_raw(diff,bend,cur);                                  // clés restantes - 1
            uint32_t nb; bool cover;
            if((diff.d[1]|diff.d[2]|diff.d[3])!=0){ nb=(uint32_t)JB; cover=false; }
            else { u64 nbr=diff.d[0]/(u64)GS+1; if(nbr>(u64)JB){ nb=(uint32_t)JB; cover=false; } else { nb=(uint32_t)nbr; cover=true; } }
            GJob& jb=jobs[nj]; memcpy(jb.lo,&cur,32); jb.nb=nb; jb.pad=0;
            jkeys[nj]=cover? diff.d[0]+1 : (u64)nb*(u64)GS;
            nj++;
            if(cover) have=false; else add_small(cur,(u64)nb*(u64)GS);
        }
        if(nj==0) break;
        // 2) lancer, confirmer
        clk::time_point t1=clk::now();
        bool ok=run_range(0,nj);
        double dt=std::chrono::duration<double>(clk::now()-t1).count();
        if(!ok){                                                                // erreur GPU : on termine ces travaux sur CPU
            if(!r.quiet) fprintf(stderr,"\n[%s] erreur GPU (%s) : bascule sur CPU.\n",r.label[tid].c_str(),err);
            for(uint32_t k=0;k<nj&&!r.stop.load();k++){
                U256 s; memcpy(&s,jobs[k].lo,32); U256 e=s; add_small(e,(u64)jobs[k].nb*(u64)GS-1);
                cpu_range(r,s,e,sc,scratch_state,-1);
            }
            gpu_fail=true; break;
        }
        u64 sum=0; for(uint32_t k=0;k<nj;k++) sum+=jkeys[k];
        st.keys.fetch_add(sum,std::memory_order_relaxed);
        // 3) travail restant de ce calculateur : le bloc en cours (le reste a été terminé)
        if(have) wip_set_one(r,tid,cur,bend); else wip_set(r,tid,std::vector<Work>());
        // 4) lancement suivant : durée visée ~0,4 s (réactivité de Ctrl+C et de l'affichage)
        if(dt>0){
            double f=0.4/dt; if(f>2.0) f=2.0; if(f<0.5) f=0.5;
            double nJ=J*f; if(nJ<256) nJ=256; if(nJ>maxjobs) nJ=maxjobs;
            J=(uint32_t)nJ;
        }
    }
    gpu_close(ctx);
    if(gpu_fail){
        if(have&&!r.stop.load()){ cpu_range(r,cur,bend,sc,scratch_state,tid); have=false; if(!r.stop.load()) wip_set(r,tid,std::vector<Work>()); }
        cpu_loop(r,g,tid,st,sc);
    }
}

static void worker(WArg* a){
    Run& r=*a->r; int tid=a->tid; TState& st=r.st[tid];
    Gen g(&r,r.mode,tid);                                                   // tous les calculateurs : le même mode
    Scratch sc;
    if(r.gdev[tid]>=0) gpu_worker(r,tid,g,st,sc); else cpu_loop(r,g,tid,st,sc);
    st.mode.store(-1);
    r.done.fetch_add(1);
}
#ifdef _WIN32
static DWORD WINAPI thread_main(LPVOID p){ worker((WArg*)p); return 0; }
static int hw_threads(){ SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors; }
static void sleep_ms(int ms){ Sleep(ms); }
#else
static int hw_threads(){ int n=(int)std::thread::hardware_concurrency(); return n>0?n:1; }
static void sleep_ms(int ms){ std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
#endif

// ---------------------------------------------------------------- reprise (checkpoint)
// Fichier : magic | en-tête (8 x u64) | bitmap des blocs pris | intervalles non terminés | clés déjà trouvées | somme de contrôle.
// Les blocs "pris" mais non terminés sont décrits par des intervalles : la reprise n'oublie rien et ne refait presque rien,
// quel que soit le nombre de cartes / cœurs de la nouvelle session.
static const char CK_MAGIC[8]={'P','Z','C','K','P','T','0','2'};
static u64 run_param_hash(const Run& r){
    u64 h=0xC0FFEE1234567ULL;
    h=sum_hash(&r.start,32,h); h=sum_hash(&r.end,32,h);
    u64 m[6]={(u64)r.mode,(u64)(r.ts->addr?1:0),(u64)r.bsize,r.NB,(u64)r.ts->list.size(),r.ts->src_hash};
    h=sum_hash(m,sizeof m,h);
    h=sum_hash(r.ts->list.data(),r.ts->list.size()*sizeof(Ent),h);
    return h;
}
static bool ckpt_save(Run& r,double el_session,u64 tested_session){
    if(r.ckpt_path.empty()) return true;
    std::vector<u64> claim; std::vector<Work> all; u64 seq;
    r.ck.lock();
    claim.resize(r.claim.size()); for(size_t i=0;i<claim.size();i++) claim[i]=r.claim[i].load();
    seq=r.seq.load();
    for(const Work& w:r.pending) all.push_back(w);
    for(const auto& v:r.wip) for(const Work& w:v) all.push_back(w);
    r.ck.unlock();
    std::vector<U256> fk; r.lock.lock(); for(const Found& f:r.found) fk.push_back(f.key); r.lock.unlock();
    u64 hdr[8]={r.ckpt_hash,(u64)claim.size(),(u64)all.size(),(u64)fk.size(),seq,
                (u64)((r.base_elapsed+el_session)*1000.0),r.base_tested+tested_session,ENDIAN_TAG};
    u64 sum=sum_hash(hdr,sizeof hdr,0x5EED5EED5EEDULL);
    sum=sum_hash(claim.data(),claim.size()*8,sum); sum=sum_hash(all.data(),all.size()*sizeof(Work),sum);
    sum=sum_hash(fk.data(),fk.size()*sizeof(U256),sum);
    std::string tmp=r.ckpt_path+".tmp";
    FILE* f=fopen(tmp.c_str(),"wb"); if(!f) return false;
    bool ok=fwrite(CK_MAGIC,1,8,f)==8&&fwrite(hdr,8,8,f)==8
        &&(claim.empty()||fwrite(claim.data(),8,claim.size(),f)==claim.size())
        &&(all.empty()||fwrite(all.data(),sizeof(Work),all.size(),f)==all.size())
        &&(fk.empty()||fwrite(fk.data(),sizeof(U256),fk.size(),f)==fk.size())
        &&fwrite(&sum,8,1,f)==1;
    if(fflush(f)!=0) ok=false;
#ifndef _WIN32
    if(ok&&fsync(fileno(f))!=0) ok=false;
#endif
    if(fclose(f)!=0) ok=false;
    if(!ok){ remove(tmp.c_str()); return false; }
#ifdef _WIN32
    remove(r.ckpt_path.c_str());
#endif
    if(rename(tmp.c_str(),r.ckpt_path.c_str())!=0){ remove(tmp.c_str()); return false; }
    return true;
}
// Retour : 1 = reprise chargée ; 0 = pas de reprise (msg dit pourquoi si le fichier existait)
static int ckpt_load(Run& r,std::string& msg){
    msg.clear();
    std::string d; if(r.ckpt_path.empty()||!read_file(r.ckpt_path,d)) return 0;
    const size_t HB=8+8*8;
    if(d.size()<HB+8||memcmp(d.data(),CK_MAGIC,8)!=0){ msg="fichier de reprise illisible (ignoré)"; return 0; }
    u64 hdr[8]; memcpy(hdr,d.data()+8,sizeof hdr);
    if(hdr[7]!=ENDIAN_TAG){ msg="fichier de reprise d'une autre architecture (ignoré)"; return 0; }
    u64 ncl=hdr[1],nw=hdr[2],nf=hdr[3];
    if(ncl!=r.claim.size()||nw>(1ULL<<32)||nf>(1ULL<<32)){ msg="fichier de reprise incompatible avec cette recherche (ignoré)"; return 0; }
    size_t need=HB+ncl*8+nw*sizeof(Work)+nf*sizeof(U256)+8;
    if(d.size()!=need){ msg="fichier de reprise tronqué (ignoré)"; return 0; }
    const char* p=d.data()+HB;
    std::vector<u64> claim(ncl); memcpy(claim.data(),p,ncl*8); p+=ncl*8;
    std::vector<Work> all(nw); if(nw) memcpy(all.data(),p,nw*sizeof(Work)); p+=nw*sizeof(Work);
    std::vector<U256> fk(nf); if(nf) memcpy(fk.data(),p,nf*sizeof(U256)); p+=nf*sizeof(U256);
    u64 sum; memcpy(&sum,p,8);
    u64 s2=sum_hash(hdr,sizeof hdr,0x5EED5EED5EEDULL);
    s2=sum_hash(claim.data(),claim.size()*8,s2); s2=sum_hash(all.data(),all.size()*sizeof(Work),s2);
    s2=sum_hash(fk.data(),fk.size()*sizeof(U256),s2);
    if(s2!=sum){ msg="fichier de reprise corrompu (ignoré)"; return 0; }
    if(hdr[0]!=r.ckpt_hash){ msg="le fichier de reprise correspond à une autre recherche (ignoré)"; return 0; }
    for(size_t i=0;i<claim.size();i++) r.claim[i].store(claim[i]);
    r.seq.store(hdr[4]);
    for(const Work& w:all){                                        // garde-fou : intervalle valide et dans la plage
        if(geq(w.e,w.s)&&geq(w.s,r.start)&&geq(r.end,w.e)) r.pending.push_back(w);
    }
    r.base_elapsed=hdr[5]/1000.0; r.base_tested=hdr[6];
    for(const U256& k:fk){                                          // clés déjà trouvées (déjà dans le fichier de résultats)
        Aff P=aff_mul(k,GEN); if(P.inf) continue;
        Ent e=ent_of_point(r.ts->addr,P);
        if(!ts_has_exact(*r.ts,e.x,e.par)) continue;
        Found f; f.key=k; f.ent=e; f.pub=compress_hex(P); f.t=0; f.saved=true; f.path="(session précédente)";
        f.outside=!geq(r.end,k)||!geq(k,r.start);
        r.found.push_back(f); r.nfound.fetch_add(1);
    }
    return 1;
}

// ---------------------------------------------------------------- affichage
static bool g_ansi=false;
static bool g_plain=false;                 // hors terminal (Colab, Kaggle, redirection) : une ligne de statut toutes les g_log_s secondes
static double g_log_s=10.0;
static std::atomic<bool> g_sigint(false);
static void on_sigint(int){ g_sigint.store(true); }
static const char* CL(const char* c){ return g_ansi?c:""; }
static const int INNER=74;
static int vis_len(const std::string& s){
    int n=0;
    for(size_t i=0;i<s.size();){
        unsigned char c=s[i];
        if(c==0x1b){ while(i<s.size()&&s[i]!='m') i++; i++; continue; }
        if((c&0xC0)!=0x80) n++;
        i++;
    }
    return n;
}
static std::string rep(const char* s,int n){ std::string o; for(int i=0;i<n;i++) o+=s; return o; }
static std::string row(const std::string& c){
    std::string o=std::string(CL("\x1b[90m"))+"║"+CL("\x1b[0m")+" "+c;
    int v=vis_len(c); if(v<INNER) o.append(INNER-v,' ');
    o+=std::string(" ")+CL("\x1b[90m")+"║"+CL("\x1b[0m")+"\n";
    return o;
}
static std::string border(const char* l,const char* m,const char* rr){
    return std::string(CL("\x1b[90m"))+l+rep(m,INNER+2)+rr+CL("\x1b[0m")+"\n";
}
static std::string padr(const std::string& s,int w){
    std::string o=s; int v=vis_len(s); if(v<w) o.append(w-v,' '); return o;
}
static const char* MODE_COL[MD_COUNT]={"\x1b[36m","\x1b[35m","\x1b[34m","\x1b[32m","\x1b[33m","\x1b[95m","\x1b[96m"};

struct Snap {
    double el=0; u64 tested=0; double speed=0;
    std::vector<double> tsp; std::vector<Found> found;
};
static std::string build_frame(Run& r,const Snap& s){
    std::string f;
    const TargetSet& ts=*r.ts;
    long double total=r.total;
    long double pct=100.0L*s.tested/total; if(pct>100) pct=100;
    double avg=s.speed;                                   // vitesse moyenne de la session courante
    long double eta=avg>0?(total-s.tested)/avg:-1;            // -1 : vitesse pas encore connue -> "?"
    char b[256];

    f+=border("╔","═","╗");
    f+=row(std::string(CL("\x1b[1m\x1b[36m"))+"PUZZLE BITCOIN · secp256k1 · recherche multi-GPU / multi-cœurs"+CL("\x1b[0m"));
    f+=border("╠","═","╣");
    double mb=(ts.bloom.size()*8.0+ts.list.size()*sizeof(Ent))/1048576.0;
    snprintf(b,sizeof b,"Cibles  : %s %s  (filtre + liste : %.1f Mo)",fmt_int(ts.list.size()).c_str(),ts.addr?"adresse(s)":"clé(s) publique(s)",mb);
    f+=row(b);
    int ngpu=0; for(int g:r.gdev) if(g>=0) ngpu++;
    if(r.is_range) snprintf(b,sizeof b,"Taille  : %s clés   Blocs : %s × 2^%d   GPU : %d   CPU : %d",fmt_keys(r.total).c_str(),fmt_int(r.NB).c_str(),r.bsize,ngpu,r.nthreads-ngpu);
    else snprintf(b,sizeof b,"Bits    : %d   Blocs : %s × 2^%d clés   GPU : %d   CPU : %d",r.bits,fmt_int(r.NB).c_str(),r.bsize,ngpu,r.nthreads-ngpu);
    f+=row(b);
    f+=row("Plage   : "+short_hex(r.start)+"  →  "+short_hex(r.end));
    f+=row(std::string("Mode    : ")+MODE_NAME[r.mode]+"  (tous les calculateurs sur la même suite de blocs)");
    if(r.base_elapsed>0) f+=row(std::string(CL("\x1b[33m"))+"Reprise : session précédente "+fmt_time(r.base_elapsed)+CL("\x1b[0m"));
    f+=border("╠","═","╣");
    int shown=std::min(r.nthreads,12);
    for(int i=0;i<shown;i++){
        int m=r.st[i].mode.load(); u64 blk=r.st[i].block.load();
        std::string col=(m>=0&&g_ansi)?MODE_COL[m]:"";
        U256 bs=r.start; add_raw(bs,bs,shl_u64(blk,r.bsize));
        std::string lb=r.label[i]; if((int)lb.size()>18) lb=lb.substr(0,18);
        std::string line=col+padr(lb,18)+CL("\x1b[0m")+" ";
        snprintf(b,sizeof b,"bloc %9s  ",m>=0?fmt_int(blk).c_str():"terminé");
        line+=b;
        line+=padr(m>=0?short_hex(bs):"-",24);
        snprintf(b,sizeof b,"%9.2f Mk/s",s.tsp[i]/1e6);
        line+=b;
        f+=row(line);
    }
    if(r.nthreads>shown){ snprintf(b,sizeof b,"… +%d autres calculateurs",r.nthreads-shown); f+=row(CL("\x1b[90m")+std::string(b)+CL("\x1b[0m")); }
    f+=border("╠","═","╣");
    int W=40; int fl=(int)(pct/100.0L*W+0.0001L); if(fl>W) fl=W;
    snprintf(b,sizeof b,"  %6.2Lf %%",pct);
    f+=row(std::string(CL("\x1b[32m"))+rep("█",fl)+CL("\x1b[90m")+rep("░",W-fl)+CL("\x1b[0m")+b);
    snprintf(b,sizeof b,"Testé   : %-12s Vitesse : %s clés/s",fmt_keys((long double)s.tested).c_str(),fmt_keys((long double)s.speed).c_str());
    f+=row(b);
    snprintf(b,sizeof b,"Écoulé  : %-12s Reste   : ~%s",fmt_time(s.el).c_str(),fmt_time(eta).c_str());
    f+=row(b);
    snprintf(b,sizeof b,"Trouvées: %zu / %s",s.found.size(),fmt_int(ts.list.size()).c_str());
    f+=row(std::string(s.found.empty()?"":CL("\x1b[1m\x1b[32m"))+b+CL("\x1b[0m"));
    f+=border("╠","═","╣");
    int fstart=std::max(0,(int)s.found.size()-3);
    for(int i=0;i<3;i++){
        int idx=fstart+i;
        if(idx>=0&&idx<(int)s.found.size()){
            const Found& fd=s.found[idx];
            std::string pk=ent_text(ts,fd.ent); if(fd.outside) pk+=" (hors plage)"; pk=pk.substr(0,10)+"…"+pk.substr(pk.size()-6);
            std::string mark=fd.saved ? std::string(CL("\x1b[32m"))+"  ✔ enregistrée"+CL("\x1b[0m")
                                      : std::string(CL("\x1b[1m\x1b[31m"))+"  ⚠ NON ENREGISTRÉE"+CL("\x1b[0m");
            f+=row(std::string(CL("\x1b[32m"))+"★ "+CL("\x1b[0m")+pk+"  →  "+short_hex(fd.key)+mark);
        } else f+=row(i==0&&s.found.empty()?std::string(CL("\x1b[90m"))+"(aucune clé trouvée pour l'instant)"+CL("\x1b[0m"):"");
    }
    f+=border("╚","═","╝");
    return f;
}

static void run_engine(Run& r,bool display){
    r.t0=clk::now();
    int n=r.nthreads;
    std::vector<WArg> args(n);
    for(int i=0;i<n;i++){ args[i].r=&r; args[i].tid=i; }
#ifdef _WIN32
    std::vector<HANDLE> hs(n);
    for(int i=0;i<n;i++) hs[i]=CreateThread(NULL,0,thread_main,&args[i],0,NULL);
#else
    std::vector<std::thread> ths;
    for(int i=0;i<n;i++) ths.emplace_back(worker,&args[i]);
#endif
    struct Samp { double t; u64 k; };
    std::vector<std::deque<Samp>> hist(n);                     // vitesse lissée sur ~5 s (les lancements GPU sont par à-coups)
    int prevLines=0; size_t shown=0; int lastnf=0; double lastplain=-1e9;
    auto elapsed=[&](){ return std::chrono::duration<double>(clk::now()-r.t0).count(); };
    auto session_tested=[&]()->u64{ u64 t=0; for(int i=0;i<n;i++) t+=r.st[i].keys.load(); return t; };
    auto draw=[&](bool final_){
        double el=elapsed();
        Snap s; s.tsp.assign(n,0.0);
        u64 sess=0;
        for(int i=0;i<n;i++){
            u64 k=r.st[i].keys.load(); sess+=k;
            auto& h=hist[i]; h.push_back(Samp{el,k});
            while(h.size()>2&&el-h[1].t>=5.0) h.pop_front();
            if(h.back().t>h.front().t) s.tsp[i]=(double)(h.back().k-h.front().k)/(h.back().t-h.front().t);
        }
        s.tested=r.base_tested+sess; s.el=r.base_elapsed+el; s.speed=sess/(el>0?el:1e-9);
        r.lock.lock(); s.found=r.found; r.lock.unlock();
        if(!g_ansi){                                           // sans couleurs : annonce immédiate de chaque clé trouvée
            for(size_t i=shown;i<s.found.size();i++)
                printf("\n*** CLÉ TROUVÉE ***  clé privée : %s\n    %s\n",to_hex(s.found[i].key).c_str(),
                       s.found[i].saved?("enregistrée dans "+s.found[i].path).c_str():"!!! NON ENREGISTRÉE !!!");
            if(s.found.size()>shown) fflush(stdout);
        }
        if(g_ansi){
            std::string f=build_frame(r,s);
            std::string out;
            if(s.found.size()>shown) out+="\a";                // signal sonore à chaque nouvelle clé
            if(prevLines>0){ char c[16]; snprintf(c,sizeof c,"\x1b[%dA",prevLines); out+=c; }
            out+=f; fputs(out.c_str(),stdout); fflush(stdout);
            prevLines=(int)std::count(f.begin(),f.end(),'\n');
        } else if(g_plain){                                    // journal ligne par ligne (notebooks, redirection)
            if(final_||el-lastplain>=g_log_s){
                lastplain=el;
                long double total=r.total, pct=100.0L*s.tested/total; if(pct>100) pct=100;
                long double eta=s.speed>0?(total-s.tested)/s.speed:-1;
                std::string per; int ng=0;
                for(int i=0;i<n;i++){ char b[64]; snprintf(b,sizeof b,"%s%s %s/s",per.empty()?"":" | ",r.gdev[i]>=0?("GPU"+std::to_string(r.gdev[i])).c_str():"CPU",fmt_keys(s.tsp[i]).c_str()); if(i<8) per+=b; else ng++; }
                printf("[%6.2Lf%%] %s clés/s (%s%s) | écoulé %s | reste ~%s | trouvées %zu/%zu\n",
                    pct,fmt_keys((long double)s.speed).c_str(),per.c_str(),ng?" …":"",fmt_time(s.el).c_str(),fmt_time(eta).c_str(),
                    s.found.size(),r.ts->list.size());
                fflush(stdout);
            }
        } else if(!final_){
            long double total=r.total;
            printf("\r[%6.2Lf%%] %.2f Mk/s | écoulé %s | trouvées %zu/%zu   ",
                100.0L*s.tested/total,s.speed/1e6,fmt_time(s.el).c_str(),s.found.size(),r.ts->list.size());
            fflush(stdout);
        }
        shown=s.found.size();
    };
    if(display&&g_ansi){ fputs("\x1b[?25l",stdout); }
    if(display) draw(false);
    auto tp=clk::now(); double lastck=0;
    while(r.done.load()<n){
        sleep_ms(display?100:2);
        if(g_sigint.load()) r.stop.store(true);
        if(r.auto_stop_s>=0&&elapsed()>=r.auto_stop_s) r.stop.store(true);
        int nf=r.nfound.load();
        if(display&&(nf!=lastnf||std::chrono::duration<double>(clk::now()-tp).count()>=0.5)){ lastnf=nf; tp=clk::now(); draw(false); }
        if(!r.ckpt_path.empty()){
            double el=elapsed();
            if(el-lastck>=r.ckpt_every){ lastck=el; if(!ckpt_save(r,el,session_tested())&&!r.quiet) fprintf(stderr,"\nAttention : sauvegarde de reprise impossible (%s)\n",r.ckpt_path.c_str()); }
        }
    }
#ifdef _WIN32
    for(int i=0;i<n;i++){ WaitForSingleObject(hs[i],INFINITE); CloseHandle(hs[i]); }
#else
    for(auto& t:ths) t.join();
#endif
    if(display){
        draw(true);
        if(g_ansi) fputs("\x1b[?25h",stdout); else if(!g_plain) fputs("\n",stdout);
        fflush(stdout);
    }
    if(!r.ckpt_path.empty()){
        bool interrupted=r.stop.load()&&r.found.size()<r.ts->list.size();   // arrêt demandé avant la fin : on garde la reprise
        if(interrupted){
            if(ckpt_save(r,elapsed(),session_tested())){ if(display) printf("Progression sauvegardée : %s (relance la même commande pour reprendre)\n",r.ckpt_path.c_str()); }
            else fprintf(stderr,"Attention : sauvegarde de reprise impossible (%s)\n",r.ckpt_path.c_str());
        } else remove(r.ckpt_path.c_str());
    }
}

// ---------------------------------------------------------------- initialisation & auto-tests
static void init_curve(){
    parse_hex(GX_HEX,GEN.x); parse_hex(GY_HEX,GEN.y);
    GEN.inf=false;
}

static bool engine_test(int bits,const std::vector<u64>& keys,const std::vector<u64>& expect,int nthreads,bool addr=false,int mode=MD_RANDOM){
    TargetSet ts; ts.addr=addr;
    for(u64 k:keys){ U256 kk={{k,0,0,0}}; Aff P=aff_mul(kk,GEN); ts.list.push_back(ent_of_point(addr,P)); }
    finalize_targets(ts);
    Run r; init_run(r,bits,nthreads,&ts,false,mode);
    run_engine(r,false);
    if(r.found.size()!=expect.size()) return false;
    for(u64 k:expect){
        bool ok=false;
        for(const Found& f:r.found) if(f.key.d[0]==k&&!f.key.d[1]&&!f.key.d[2]&&!f.key.d[3]) ok=true;
        if(!ok) return false;
    }
    return true;
}


static bool range_engine_test(u64 a,u64 b,const std::vector<u64>& keys,const std::vector<u64>& expect,int nthreads,bool addr,int mode){
    TargetSet ts; ts.addr=addr;
    for(u64 k:keys){ U256 kk={{k,0,0,0}}; ts.list.push_back(ent_of_point(addr,aff_mul(kk,GEN))); }
    finalize_targets(ts);
    Run r; U256 sa={{a,0,0,0}}, sb={{b,0,0,0}}; init_run_range(r,sa,sb,nthreads,&ts,false,mode);
    run_engine(r,false);
    if(r.found.size()!=expect.size()) return false;
    for(u64 k:expect){ bool ok=false; for(const Found& f:r.found) if(f.key.d[0]==k&&!f.key.d[1]&&!f.key.d[2]&&!f.key.d[3]) ok=true; if(!ok) return false; }
    return true;
}
static bool range_coverage_test(u64 a,u64 b,int mode,int ncores){
    TargetSet ts; { Aff P=aff_mul(ONE,GEN); ts.list.push_back(ent_of_point(false,P)); } finalize_targets(ts);
    Run r; U256 sa={{a,0,0,0}}, sb={{b,0,0,0}}; init_run_range(r,sa,sb,ncores,&ts,false,mode);
    std::vector<Gen> gens; for(int t=0;t<ncores;t++) gens.emplace_back(&r,mode,t);
    std::vector<char> seen(r.NB,0); u64 got=0; std::vector<char> alive(ncores,1); int live=ncores;
    for(int t=0;live>0;t=(t+1)%ncores){
        if(!alive[t]) continue;
        u64 blk; if(!gens[t].next(blk)){ alive[t]=0; live--; continue; }
        if(blk>=r.NB||seen[blk]) return false;
        seen[blk]=1; got++;
    }
    // le dernier bloc doit se terminer exactement à la fin de la plage
    U256 s2=r.start; add_raw(s2,s2,shl_u64(r.NB-1,r.bsize)); U256 e2; add_raw(e2,s2,r.blockmask);
    return got==r.NB && geq(e2,r.end) && geq(r.end,s2);
}

// Chaque mode doit donner chaque bloc une fois et une seule (plusieurs "cœurs" simulés en alternance).
static bool coverage_test(int bits,int mode,int ncores){
    TargetSet ts; { U256 k=ONE; Aff P=aff_mul(k,GEN); ts.list.push_back(ent_of_point(false,P)); } finalize_targets(ts);
    Run r; init_run(r,bits,ncores,&ts,false,mode);
    std::vector<Gen> gens; for(int t=0;t<ncores;t++) gens.emplace_back(&r,mode,t);
    std::vector<char> seen(r.NB,0); u64 got=0; std::vector<char> alive(ncores,1); int live=ncores;
    for(int t=0;live>0;t=(t+1)%ncores){
        if(!alive[t]) continue;
        u64 b; if(!gens[t].next(b)){ alive[t]=0; live--; continue; }
        if(b>=r.NB||seen[b]) return false;
        seen[b]=1; got++;
    }
    return got==r.NB;
}

// Enregistrement : écriture + relecture, copie de secours, panne d'un fichier, panne totale (sans plantage)
static bool save_test(){
    bool ok=true;
    TargetSet ts; U256 k={{0xABCDE,0,0,0}}; ts.list.push_back(ent_of_point(false,aff_mul(k,GEN))); finalize_targets(ts);
    const char* fa="selftest_a.tmp"; const char* fb="selftest_b.tmp";
    remove(fa); remove(fb);
    {   // 1) deux fichiers valides : les deux contiennent la clé
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main=fa; r.out_copy=fb; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        std::string ca,cb; read_file(fa,ca); read_file(fb,cb);
        std::string hex=to_hex(k);
        ok=ok&&r.found.size()==1&&r.found[0].saved&&ca.find(hex)!=std::string::npos&&cb.find(hex)!=std::string::npos;
    }
    remove(fa); remove(fb);
    {   // 2) fichier principal impossible : la copie sauve la clé
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main="/dossier_inexistant_xyz/a.tmp"; r.out_copy=fb; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        std::string cb; read_file(fb,cb);
        ok=ok&&r.found.size()==1&&r.found[0].saved&&cb.find(to_hex(k))!=std::string::npos;
    }
    remove(fb);
    {   // 3) aucun fichier possible : signalé comme NON enregistré, sans plantage
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main="/dossier_inexistant_xyz/a.tmp"; r.out_copy="/dossier_inexistant_xyz/b.tmp"; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        ok=ok&&r.found.size()==1&&!r.found[0].saved;
    }
    return ok;
}

static bool self_test(){
    bool ok=true;
    auto check=[&](bool c,const char* name){ if(!c){ fprintf(stderr,"AUTO-TEST ÉCHEC : %s\n",name); ok=false; } };
    check(on_curve(GEN),"G sur la courbe");
    U256 ex;
    Aff g2=aff_mul({{2,0,0,0}},GEN); parse_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",ex);
    check(eq(g2.x,ex),"2G");
    Aff g3=aff_mul({{3,0,0,0}},GEN); parse_hex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",ex);
    check(eq(g3.x,ex),"3G");
    U256 ks[4]={{{0xDEADBEEFCAFEBABEULL,0x0123456789ABCDEFULL,0x1111222233334444ULL,0x0ABCDEF012345678ULL}},
                {{0xFFFFFFFFFFFFFFFFULL,0,0,0}},{{0,1,0,0}},{{12345,0,0xABCDULL,0}}};
    for(int i=0;i<4;i++){
        Aff a=aff_mul(ks[i],GEN), j=jac_to_aff(jac_mul(ks[i],GEN));
        check(eq(a.x,j.x)&&eq(a.y,j.y)&&on_curve(a),"affine == jacobien");
    }
    build_tables();
    for(int i=1;i<=HALF;i++){
        Aff j=jac_to_aff(jac_mul({{(u64)i,0,0,0}},GEN));
        if(!(eq(j.x,gx[i])&&eq(j.y,gy[i]))){ check(false,"table i*G"); break; }
    }
    Aff ng=jac_to_aff(jac_mul({{(u64)(2*HALF),0,0,0}},GEN));
    check(eq(ng.x,SST.x)&&eq(ng.y,SST.y),"pas 2*HALF*G");

    // arithmétique : carré dédié, inversion par chaîne, hash160 par paquets
    {
        Rng g; g.seed(777); int bad=0;
        for(int n=0;n<4000;n++){
            U256 a; for(int i=0;i<4;i++) a.d[i]=g.next();
            if(n%5==1){ a=PRIME; a.d[0]-=g.next()%3000; }
            if(n%7==2){ a=ZERO; a.d[0]=g.next()%5; }
            if(geq(a,PRIME)) sub_raw(a,a,PRIME);
            U256 s1,s2; fe_sqr(s1,a); fe_mul(s2,a,a); if(!eq(s1,s2)) bad++;
            if(n%40==0&&!is_zero(a)){
                U256 inv=fe_inv(a),pr; fe_mul(pr,inv,a); if(!eq(pr,ONE)) bad++;
                U256 e=PRIME; e.d[0]-=2; if(!eq(inv,fe_pow(a,e))) bad++;
            }
        }
        check(bad==0,"carré / inversion");
        int bh=0;
        for(int n=0;n<200;n++){
            U256 xs[VLANES],hs[VLANES]; u64 ps[VLANES];
            for(int l=0;l<VLANES;l++){ for(int i=0;i<4;i++) xs[l].d[i]=g.next(); ps[l]=g.next()&1; }
            hash160_vec(xs,ps,hs);
            for(int l=0;l<VLANES;l++) if(!eq(hs[l],hash160_fast(xs[l],ps[l]))) bh++;
        }
        check(bh==0,"hash160 par paquets");
    }
    // SHA-256 / RIPEMD-160 / adresses (vecteurs connus : clé privée 1)
    {
        unsigned char pb[33],h[20]; pub33(GEN.x,GEN.y.d[0]&1,pb);
        hash160_bytes(pb,33,h);
        U256 want; parse_hex("00000000751e76e8199196d454941c45d1b3a323f1433bd6",want);
        unsigned char wb[20]={0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
        check(memcmp(h,wb,20)==0,"hash160(G) octets");
        check(eq(hash160_fast(GEN.x,GEN.y.d[0]&1),pack20(wb)),"hash160(G) rapide");
        check(eq(hash160_fast(g2.x,g2.y.d[0]&1),hash160_point_bytes(g2)),"hash160 rapide == octets (2G)");
        check(eq(hash160_fast(g3.x,g3.y.d[0]&1),hash160_point_bytes(g3)),"hash160 rapide == octets (3G)");
        (void)want;
        Ent e; e.x=pack20(wb); e.par=0;
        check(ent_addr_legacy(e)=="1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH","encodage Base58Check");
        check(ent_addr_bech32(e)=="bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4","encodage Bech32");
        Ent d1,d2,d3; 
        check(parse_address("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",d1)==1&&eq(d1.x,e.x),"décodage Base58Check");
        check(parse_address("BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4",d2)==1&&eq(d2.x,e.x),"décodage Bech32 (majuscules)");
        check(parse_address("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMI",d3)==0,"checksum Base58 invalide");
        check(parse_address("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5",d3)==0,"checksum Bech32 invalide");
        check(parse_address("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",d3)==2,"P2SH non supporté");
        check(parse_address("bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0",d3)==2,"Taproot non supporté");
    }
    // lecture de texte : doublons, lignes invalides, CRLF, majuscules
    {
        Aff p1=aff_mul({{5,0,0,0}},GEN), p2=aff_mul({{77,0,0,0}},GEN);
        std::string a=compress_hex(p1), b=compress_hex(p2), A=a;
        for(auto& c:A) if(c>='a'&&c<='f') c-=32;
        std::string txt=a+"\r\n\r\n"+A+"\n"+b+" # commentaire\nzzzz\n"+std::string("04")+a.substr(2)+"\n"+a.substr(0,66-1)+"\n";
        TargetSet ts; u64 l,rj,dp,un; parse_targets_text(txt,ts,false,l,rj,dp,un);
        check(ts.list.size()==2&&dp==1&&rj==3&&l==6,"lecture du fichier texte");
        // aller-retour du fichier .bloom
        ts.src_hash=12345;
        const char* tmp="selftest.tmp.bloom";
        if(save_bloom(ts,tmp)){
            TargetSet t2; std::string err;
            bool lo=load_bloom(tmp,t2,err);
            check(lo&&t2.list.size()==2&&t2.src_hash==12345&&t2.nblocks==ts.nblocks,"relecture .bloom");
            if(lo) check(ts_has_exact(t2,p1.x,p1.y.d[0]&1)&&ts_has_exact(t2,p2.x,p2.y.d[0]&1)&&!ts_has_x(t2,g2.x),"contenu .bloom");
            // corruption volontaire : doit être détectée
            FILE* f=fopen(tmp,"r+b");
            if(f){ fseek(f,-3,SEEK_END); int c=fgetc(f); fseek(f,-3,SEEK_END); fputc(c^0x55,f); fclose(f);
                   TargetSet t3; std::string e2; check(!load_bloom(tmp,t3,e2),"détection de corruption"); }
            remove(tmp);
        }
    }
    // recherches complètes sur des clés connues (cas dégénérés, multi-cibles, multi-threads)
    check(engine_test(1,{1},{1},1),"bits=1");
    check(engine_test(2,{3},{3},1),"bits=2");
    check(engine_test(3,{4,7,5},{4,7,5},2),"bits=3");
    check(engine_test(7,{100},{100},1),"bits=7");
    check(engine_test(10,{700,600,1000,5},{700,600,1000},1),"bits=10 multi-cibles (1 thread)");
    check(engine_test(10,{700,600,1000,5},{700,600,1000},4),"bits=10 multi-cibles (4 threads)");
    check(engine_test(11,{1031,2047},{1031,2047},3),"bits=11");
    check(engine_test(17,{70000,66000,131071},{70000,66000,131071},4),"bits=17");
    check(engine_test(20,{0xFFFFF,0x80000,0x9ABCD},{0xFFFFF,0x80000,0x9ABCD},4),"bits=20");
    check(engine_test(22,{3000001},{3000001},2),"bits=22");
    check(engine_test(16,{12345},{},1),"clé absente de la plage");
    check(save_test(),"enregistrement fiable des résultats");
    {   // plages quelconques (taille non puissance de 2)
        bool cv=true;
        for(int m=0;m<MD_COUNT;m++) for(int nc:{1,3})
            for(auto pr:std::vector<std::pair<u64,u64>>{{1,1},{5,5},{3,70},{1000,7777},{65536,65536+100000},{12345,12345+2000000},{77,(1ULL<<25)+99}})
                cv=cv&&range_coverage_test(pr.first,pr.second,m,nc);
        check(cv,"couverture exacte de plages quelconques");
        bool ok=true;
        for(int m=0;m<MD_COUNT;m++) for(bool ad:{false,true}){
            ok=ok&&range_engine_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},2,ad,m);      // 2 est hors plage
            ok=ok&&range_engine_test(5,5,{5},{5},1,ad,m);
            ok=ok&&range_engine_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},3,ad,m);
        }
        check(ok,"recherche dans des plages quelconques");
    }
    for(int m=0;m<MD_COUNT;m++){                     // chaque mode : couverture exacte des blocs + recherche réelle
        bool cv=true;
        for(int b:{1,2,5,7,10,14,19,27,33,40}) for(int nc:{1,3,4}) cv=cv&&coverage_test(b,m,nc);
        check(cv,(std::string("couverture des blocs, mode ")+MODE_NAME[m]).c_str());
        check(engine_test(14,{9000,12000,16383,8192},{9000,12000,16383,8192},3,false,m),"recherche, mode clé publique");
        check(engine_test(14,{9000,12000,16383,8192},{9000,12000,16383,8192},3,true,m),"recherche, mode adresse");
    }
    {   // clés aux frontières des lots de 2*HALF clés (début, milieu, fin, lot suivant)
        std::vector<u64> ek; for(u64 o:{0,1,HALF-1,HALF,HALF+1,2*HALF-1,2*HALF,2*HALF+1,3*HALF,1000,2047}) ek.push_back(2048+o);
        check(engine_test(12,ek,ek,1),"frontières de lots (clé publique)");
        check(engine_test(12,ek,ek,3,true),"frontières de lots (adresse)");
        std::vector<u64> e2; for(u64 o:{0,255,256,511,512,100000,262143}) e2.push_back(262144+o);
        check(engine_test(19,e2,e2,2),"frontières de lots, bloc plus grand");
        check(engine_test(19,e2,e2,2,true),"frontières de lots, bloc plus grand (adresse)");
    }
    check(engine_test(10,{700,600,1000,5},{700,600,1000},1,true),"adresses bits=10 (1 thread)");
    check(engine_test(14,{9000,12000,16383,300},{9000,12000,16383},4,true),"adresses bits=14 (4 threads)");
    check(engine_test(2,{3},{3},1,true),"adresses bits=2");
    return ok;
}

// ---- Auto-test GPU : recherches complètes sur des clés connues, comparées au résultat attendu.
// (le noyau GPU est ainsi validé de bout en bout sur la carte réelle : arithmétique 32 bits, hash160, Bloom, découpage en travaux)
static U256 u256_of(u64 l0,u64 l1=0,u64 l2=0,u64 l3=0){ U256 r={{l0,l1,l2,l3}}; return r; }
static U256 u256_add(U256 a,u64 v){ add_small(a,v); return a; }
static bool gpu_case(int dev,const U256& a,const U256& b,const std::vector<U256>& keys,const std::vector<U256>& expect,
                     bool addr,int mode,int jb,const char* name,bool verbose){
    TargetSet ts; ts.addr=addr;
    for(const U256& k:keys) ts.list.push_back(ent_of_point(addr,aff_mul(k,GEN)));
    finalize_targets(ts);
    Run r; init_run_range(r,a,b,1,&ts,false,mode);
    set_workers(r,std::vector<int>(1,dev),0);
    r.quiet=true;
    int old=g_gpu_jb; g_gpu_jb=jb;
    run_engine(r,false);
    g_gpu_jb=old;
    bool ok=r.found.size()==expect.size();
    for(const U256& k:expect){ bool f=false; for(const Found& x:r.found) if(eq(x.key,k)) f=true; ok=ok&&f; }
    if(!ok&&verbose) fprintf(stderr,"AUTO-TEST GPU ÉCHEC : %s (%s, mode %s, trouvées %zu / %zu attendues)\n",
                              name,addr?"adresse":"clé publique",MODE_NAME[mode],r.found.size(),expect.size());
    return ok;
}
static bool gpu_self_test(int dev,bool verbose){
    build_gpu_tables();
    const u64 GH=(u64)gpu_half(), GS=2*GH;
    bool ok=true;
    for(int ad=0;ad<2&&ok;ad++){
        bool addr=ad!=0;
        {   // plage de 2^19 clés ; clés aux frontières de lots, de travaux et de blocs
            U256 a=u256_of(0x100000), b=u256_add(a,(1u<<19)-1);
            std::vector<U256> ks; for(u64 o:{(u64)0,(u64)1,GH-1,GH,GH+1,GS-1,GS,GS+1,3*GH,(u64)1000,(u64)12345,(u64)65535,(u64)65536,(u64)(1<<19)-1}) ks.push_back(u256_add(a,o));
            ks.push_back(u256_of(0x5)); // hors plage
            std::vector<U256> ex(ks.begin(),ks.end()-1);
            ok=ok&&gpu_case(dev,a,b,ks,ex,addr,MD_SEQ,16,"frontieres",verbose);
            ok=ok&&gpu_case(dev,a,b,ks,ex,addr,MD_RANDOM,3,"frontieres (3 lots/travail)",verbose);
        }
        {   // clés de plus de 64 bits : multiplication à base fixe sur les 256 bits
            U256 a=u256_of(0x123456789ULL,0x1000000000ULL,0x7,0x0);          // ~ 2^130
            U256 b=u256_add(a,(1u<<18)-1);
            std::vector<U256> ks; for(u64 o:{(u64)0,GS-1,GS,(u64)77777,(u64)(1<<18)-1}) ks.push_back(u256_add(a,o));
            ok=ok&&gpu_case(dev,a,b,ks,ks,addr,MD_SEQ,16,"cles > 64 bits",verbose);
            U256 a2=u256_of(0xFFFFFFFFFFFFFF00ULL,0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,0x0FFFFFFFFFFFFFFFULL); // ~ 2^252, retenues partout
            U256 b2=u256_add(a2,(1u<<17)-1);
            std::vector<U256> k2; for(u64 o:{(u64)0,(u64)255,(u64)256,(u64)GS,(u64)99999}) k2.push_back(u256_add(a2,o));
            ok=ok&&gpu_case(dev,a2,b2,k2,k2,addr,MD_PI,16,"retenues 64 bits",verbose);
        }
        {   // plages minuscules / proches de 0 : travaux dégénérés refaits sur CPU
            U256 a=u256_of(1), b=u256_of(2000);
            std::vector<U256> ks; for(u64 v:{(u64)1,(u64)2,(u64)700,(u64)1999,(u64)2000}) ks.push_back(u256_of(v));
            ok=ok&&gpu_case(dev,a,b,ks,ks,addr,MD_SEQ,16,"plage 1..2000",verbose);
            ok=ok&&gpu_case(dev,u256_of(5),u256_of(5),std::vector<U256>(1,u256_of(5)),std::vector<U256>(1,u256_of
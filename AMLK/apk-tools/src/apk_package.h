/* apk_database.h - Alpine Package Keeper (APK)
 *
 * Copyright (C) 2005-2008 Natanael Copa <n@tanael.org>
 * Copyright (C) 2008 Timo Teräs <timo.teras@iki.fi>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published
 * by the Free Software Foundation. See http://www.gnu.org/ for details.
 */

#ifndef APK_PKG_H
#define APK_PKG_H

#include "apk_version.h"
#include "apk_hash.h"
#include "apk_io.h"

struct apk_database;
struct apk_name;

#define APK_SCRIPT_INVALID		-1
#define APK_SCRIPT_PRE_INSTALL		0
#define APK_SCRIPT_POST_INSTALL		1
#define APK_SCRIPT_PRE_DEINSTALL	2
#define APK_SCRIPT_POST_DEINSTALL	3
#define APK_SCRIPT_PRE_UPGRADE		4
#define APK_SCRIPT_POST_UPGRADE		5
#define APK_SCRIPT_TRIGGER		6
#define APK_SCRIPT_MAX			7

#define APK_SIGN_NONE			0
#define APK_SIGN_VERIFY			1
#define APK_SIGN_VERIFY_IDENTITY	2
#define APK_SIGN_GENERATE_V1		3
#define APK_SIGN_GENERATE		4
#define APK_SIGN_VERIFY_AND_GENERATE	5

struct apk_sign_ctx {
	int keys_fd;
	int action;
	const EVP_MD *md;
	int num_signatures;
	int control_started : 1;
	int data_started : 1;
	int has_data_checksum : 1;
	int control_verified : 1;
	int data_verified : 1;
	char data_checksum[EVP_MAX_MD_SIZE];
	struct apk_checksum identity;
	EVP_MD_CTX mdctx;

	struct {
		apk_blob_t data;
		EVP_PKEY *pkey;
		char *identity;
	} signature;
};

#define APK_DEPMASK_REQUIRE	(APK_VERSION_EQUAL|APK_VERSION_LESS|\
				 APK_VERSION_GREATER)
#define APK_DEPMASK_CONFLICT	(0)

struct apk_dependency {
	struct apk_name *name;
	int result_mask;
	char *version;
};
APK_ARRAY(apk_dependency_array, struct apk_dependency);

#define APK_IPKGF_RUN_ALL_TRIGGERS	0x00000001

struct apk_installed_package {
	struct apk_package *pkg;
	unsigned int flags;
	struct list_head installed_pkgs_list;
	struct list_head trigger_pkgs_list;
	struct hlist_head owned_dirs;
	apk_blob_t script[APK_SCRIPT_MAX];
	struct apk_string_array *triggers;
	struct apk_string_array *pending_triggers;
};

#define APK_PKG_UNINSTALLABLE		((char*) -1)

struct apk_package {
	apk_hash_node hash_node;
	unsigned repos;
	struct apk_name *name;
	char *version, *arch;
	char *url, *description, *license;
	struct apk_dependency_array *depends;
	size_t installed_size, size;
	char *filename;
	struct apk_checksum csum;
	struct apk_installed_package *ipkg;
};
APK_ARRAY(apk_package_array, struct apk_package *);

extern const char *apk_script_types[];

void apk_sign_ctx_init(struct apk_sign_ctx *ctx, int action,
		       struct apk_checksum *identity, int keys_fd);
void apk_sign_ctx_free(struct apk_sign_ctx *ctx);
int apk_sign_ctx_process_file(struct apk_sign_ctx *ctx,
			      const struct apk_file_info *fi,
			      struct apk_istream *is);
int apk_sign_ctx_parse_pkginfo_line(void *ctx, apk_blob_t line);
int apk_sign_ctx_verify_tar(void *ctx, const struct apk_file_info *fi,
			    struct apk_istream *is);
int apk_sign_ctx_mpart_cb(void *ctx, int part, apk_blob_t blob);

int apk_dep_from_blob(struct apk_dependency *dep, struct apk_database *db,
		      apk_blob_t blob);
void apk_dep_from_pkg(struct apk_dependency *dep, struct apk_database *db,
		      struct apk_package *pkg);
void apk_blob_push_dep(apk_blob_t *to, struct apk_dependency *dep);

int apk_deps_add(struct apk_dependency_array **depends,
		 struct apk_dependency *dep);
void apk_deps_del(struct apk_dependency_array **deps,
		  struct apk_name *name);
void apk_deps_parse(struct apk_database *db,
		    struct apk_dependency_array **depends,
		    apk_blob_t blob);

int apk_deps_write(struct apk_dependency_array *deps, struct apk_ostream *os);
int apk_script_type(const char *name);

void apk_pkg_format_plain(struct apk_package *pkg, apk_blob_t to);
void apk_pkg_format_cache(struct apk_package *pkg, apk_blob_t to);
struct apk_package *apk_pkg_new(void);
int apk_pkg_read(struct apk_database *db, const char *name,
		 struct apk_sign_ctx *ctx, struct apk_package **pkg);
void apk_pkg_free(struct apk_package *pkg);

int apk_pkg_parse_name(apk_blob_t apkname, apk_blob_t *name, apk_blob_t *version);

int apk_pkg_add_info(struct apk_database *db, struct apk_package *pkg,
		     char field, apk_blob_t value);

struct apk_installed_package *apk_pkg_install(struct apk_database *db, struct apk_package *pkg);
void apk_pkg_uninstall(struct apk_database *db, struct apk_package *pkg);

int apk_ipkg_add_script(struct apk_installed_package *ipkg,
			struct apk_istream *is,
			unsigned int type, unsigned int size);
int apk_ipkg_run_script(struct apk_installed_package *ipkg,
			struct apk_database *db,
			unsigned int type, char **argv);

struct apk_package *apk_pkg_parse_index_entry(struct apk_database *db, apk_blob_t entry);
int apk_pkg_write_index_entry(struct apk_package *pkg, struct apk_ostream *os, int write_arch);

int apk_pkg_version_compare(struct apk_package *a, struct apk_package *b);

#endif

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

#ifndef APK_PKGDB_H
#define APK_PKGDB_H

#include "apk_version.h"
#include "apk_hash.h"
#include "apk_archive.h"
#include "apk_package.h"
#include "apk_io.h"

#define APK_MAX_REPOS		32
#define APK_CACHE_CSUM_BYTES	4

extern const char * const apk_index_gz;
extern const char * const apkindex_tar_gz;

struct apk_name;
APK_ARRAY(apk_name_array, struct apk_name *);

struct apk_db_file {
	struct hlist_node hash_node;
	struct hlist_node diri_files_list;

	struct apk_db_dir_instance *diri;
	unsigned short namelen;
	struct apk_checksum csum;
	char name[];
};

#define APK_DBDIRF_PROTECTED		0x01
#define APK_DBDIRF_SYMLINKS_ONLY	0x02
#define APK_DBDIRF_MODIFIED		0x04

struct apk_db_dir {
	apk_hash_node hash_node;

	unsigned long hash;
	struct hlist_head files;
	struct apk_db_dir *parent;

	unsigned short refs;
	unsigned short namelen;
	unsigned char flags;
	char rooted_name[1];
	char name[];
};

struct apk_db_dir_instance {
	struct hlist_node pkg_dirs_list;
	struct hlist_head owned_files;
	struct apk_package *pkg;
	struct apk_db_dir *dir;
	mode_t mode;
	uid_t uid;
	gid_t gid;
};

#define APK_NAME_TOPLEVEL		0x0001
#define APK_NAME_REINSTALL		0x0002
#define APK_NAME_TOPLEVEL_OVERRIDE	0x0004

struct apk_name {
	apk_hash_node hash_node;
	unsigned int id;
	unsigned int flags;
	char *name;
	struct apk_package_array *pkgs;
	struct apk_name_array *rdepends;
};

struct apk_repository {
	char *url;
	struct apk_checksum csum;

	apk_blob_t description;
};

struct apk_repository_list {
	struct list_head list;
	const char *url;
};

struct apk_db_options {
	int lock_wait;
	unsigned long open_flags;
	char *root;
	char *keys_dir;
	char *repositories_file;
	struct list_head repository_list;
};

struct apk_database {
	char *root;
	int root_fd, lock_fd, cache_fd, cachetmp_fd, keys_fd;
	unsigned name_id, num_repos;
	const char *cache_dir, *arch;
	unsigned int local_repos, bad_repos;
	int permanent : 1;
	int compat_newfeatures : 1;
	int compat_notinstallable : 1;

	struct apk_dependency_array *world;
	struct apk_string_array *protected_paths;
	struct apk_repository repos[APK_MAX_REPOS];
	struct apk_id_cache id_cache;

	struct {
		struct apk_hash names;
		struct apk_hash packages;
	} available;

	struct {
		struct list_head packages;
		struct list_head triggers;
		struct apk_hash dirs;
		struct apk_hash files;
		struct {
			unsigned files;
			unsigned dirs;
			unsigned packages;
		} stats;
	} installed;
};

typedef union apk_database_or_void {
	struct apk_database *db;
	void *ptr;
} apk_database_t __attribute__ ((__transparent_union__));

struct apk_name *apk_db_get_name(struct apk_database *db, apk_blob_t name);
struct apk_name *apk_db_query_name(struct apk_database *db, apk_blob_t name);
struct apk_db_dir *apk_db_dir_query(struct apk_database *db,
				    apk_blob_t name);
struct apk_db_file *apk_db_file_query(struct apk_database *db,
				      apk_blob_t dir,
				      apk_blob_t name);

#define APK_OPENF_READ			0x0001
#define APK_OPENF_WRITE			0x0002
#define APK_OPENF_CREATE		0x0004
#define APK_OPENF_NO_INSTALLED		0x0010
#define APK_OPENF_NO_SCRIPTS		0x0020
#define APK_OPENF_NO_WORLD		0x0040
#define APK_OPENF_NO_SYS_REPOS		0x0100
#define APK_OPENF_NO_INSTALLED_REPO	0x0200

#define APK_OPENF_NO_REPOS	(APK_OPENF_NO_SYS_REPOS |	\
				 APK_OPENF_NO_INSTALLED_REPO)
#define APK_OPENF_NO_STATE	(APK_OPENF_NO_INSTALLED |	\
				 APK_OPENF_NO_SCRIPTS |		\
				 APK_OPENF_NO_WORLD)

int apk_db_open(struct apk_database *db, struct apk_db_options *dbopts);
void apk_db_close(struct apk_database *db);
int apk_db_write_config(struct apk_database *db);
int apk_db_run_triggers(struct apk_database *db);
int apk_db_permanent(struct apk_database *db);

struct apk_package *apk_db_pkg_add(struct apk_database *db, struct apk_package *pkg);
struct apk_package *apk_db_get_pkg(struct apk_database *db, struct apk_checksum *csum);
struct apk_package *apk_db_get_file_owner(struct apk_database *db, apk_blob_t filename);

int apk_db_index_read(struct apk_database *db, struct apk_bstream *bs, int repo);
int apk_db_index_read_file(struct apk_database *db, const char *file, int repo);
int apk_db_index_write(struct apk_database *db, struct apk_ostream *os);

int apk_db_add_repository(apk_database_t db, apk_blob_t repository);
struct apk_repository *apk_db_select_repo(struct apk_database *db,
					  struct apk_package *pkg);
int apk_repository_update(struct apk_database *db, struct apk_repository *repo);

int apk_db_cache_active(struct apk_database *db);
void apk_cache_format_index(apk_blob_t to, struct apk_repository *repo, int ver);
int apk_cache_download(struct apk_database *db, const char *arch, const char *url,
		       const char *item, const char *cache_item, int verify);

int apk_db_install_pkg(struct apk_database *db,
		       struct apk_package *oldpkg,
		       struct apk_package *newpkg,
		       apk_progress_cb cb, void *cb_ctx);

#endif
